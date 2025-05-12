# General Library Imports
from collections import deque, Counter
from queue import Queue
from threading import Thread
import numpy as np
from PIL import Image
import time
from collections import deque
import matplotlib.pyplot as plt
# PyQt imports
from PySide2.QtCore import QThread, Signal
import pyqtgraph as pg
from tensorflow.keras.models import load_model
import tensorflow as tf

# identical to the previous one
# Local Imports
from gui_parser import UARTParser
from gui_common import *
from graph_utilities import *

# Logger
import logging
log = logging.getLogger(__name__)

# Classifier Configurables
MAX_NUM_TRACKS = 20  # This could vary depending on the configuration file. Use 20 here as a safe likely maximum to ensure there's enough memory for the classifier

# Expected minimums and maximums to bound the range of colors used for coloring points
SNR_EXPECTED_MIN = 5
SNR_EXPECTED_MAX = 40
SNR_EXPECTED_RANGE = SNR_EXPECTED_MAX - SNR_EXPECTED_MIN
DOPPLER_EXPECTED_MIN = -30
DOPPLER_EXPECTED_MAX = 30
DOPPLER_EXPECTED_RANGE = DOPPLER_EXPECTED_MAX - DOPPLER_EXPECTED_MIN

# Different methods to color the points
COLOR_MODE_SNR = 'SNR'
COLOR_MODE_HEIGHT = 'Height'
COLOR_MODE_DOPPLER = 'Doppler'
COLOR_MODE_TRACK = 'Associated Track'

# Magic Numbers for Target Index TLV
TRACK_INDEX_WEAK_SNR = 253  # Point not associated, SNR too weak
TRACK_INDEX_BOUNDS = 254  # Point not associated, located outside boundary of interest
TRACK_INDEX_NOISE = 255  # Point not associated, considered as noise


class parseUartThread(QThread):
    fin = Signal(dict)

    def __init__(self, uParser, window_size=30, stride=2,viewer=None):
        QThread.__init__(self)
        self.parser = uParser
        self.queue = Queue()  # 🧵 buat antrian
        self.last_candidate_label = None
        self.candidate_count = 0
        self.current_label_idx = None
        self.pred_queue = deque(maxlen=5)  # history prediksi
        self.conf_queue = deque(maxlen=5)  # history confidence
        self.predThread = Thread(target=self.prediction_thread_func)
        self.predThread.daemon = True
        self.predThread.start()
        self.model = load_model("andrikun_fixed.h5", compile=False)
        self.class_names = ["Berdiri", "Duduk", "jalan", "Jatuh"]
        self.init_prediction_logger()
        self.prediction_log_buffer = []

        # 🔥 Tambahkan ini untuk logika stabilisasi jatuh
        self.last_label_name = None
        self.last_jatuh_timestamp = None
        self.doppler_change_threshold = 0.5  # Threshold mean doppler untuk "bergerak" lagi
        self.jatuh_hold_time = 3  # Detik mempertahankan status jatuh
        self.frameBuffer = deque(maxlen=window_size)
        self.window_size = window_size
        self.stride = stride
        self.counter = 0
        self.timestamp = time.strftime("%m%d%Y%H%M%S")
        self.outputDir = f'./dataset/{self.timestamp}'
        # Ensure the directory is created only once
        os.makedirs(self.outputDir, exist_ok=True)

    def run(self):
        if not hasattr(self, 'logger_initialized'):
            self.init_prediction_logger()
            self.logger_initialized = True
            
        if self.parser.parserType == "SingleCOMPort":
            outputDict = self.parser.readAndParseUartSingleCOMPort()
        else:
            outputDict = self.parser.readAndParseUartDoubleCOMPort()

            frameJSON = {
                'frameData': outputDict,
                'timestamp': time.time() * 1000
            }
        self.fin.emit(outputDict)
        # print("Emit")
        # Tambah ke buffer (deque otomatis geser kalau penuh)
        self.frameBuffer.append(frameJSON)

        # Sliding: proses hanya kalau buffer sudah penuh dan sesuai stride
        if len(self.frameBuffer) == self.window_size:
            if self.counter % self.stride == 0:
                self.process_window(list(self.frameBuffer))  # konversi deque ke list
            self.counter += 1

            # Thread 2: prediksi

    def process_window(self, frameList):
        os.makedirs("debug", exist_ok=True)
        all_points = [self.extract_features(frame) for frame in frameList]
        result = np.vstack(all_points)

        df_subset = {
            "timestamp": result[:, 0],
            "Range": result[:, 1],
            "doppler": result[:, 2],
            "SNR": result[:, 3],
        }

        dr, dt, rt, heatmap_rgb = self.generate_rgb_heatmap_tensor(df_subset)

        # ✅ Update heatmap ke GUI tetap di thread ini
        if hasattr(self, 'guiWindow'):
            self.guiWindow.updateHeatmapGUI(dr, dt, rt)

        # ✅ Kirim heatmap_rgb ke queue untuk diprediksi di thread lain
        self.queue.put((heatmap_rgb, df_subset["doppler"]))  # 🔥 Kirim heatmap dan doppler
            # self.guiWindow.updateVoxelGUI(voxel)

        # heatmap_rgb = self.generate_rgb_heatmap_tensor(df_subset)
        # if hasattr(self, 'guiWindow'):
        #     self.guiWindow.updateHeatmapGUI(heatmap_rgb)
        #     return dr, dt, rt


        # # 🔍 Simpan gambar heatmap untuk dicek visual (tanpa show)
        # start_ts = int(frameList[0]['timestamp'])
        # self.debug_visualize_heatmap(heatmap_rgb, start_ts)
        # if self.viewer:
        #     self.viewer.update(heatmap_rgb)
        # ✅ Kirim ke model kalau ada
        # prediction = self.model.predict(np.expand_dims(heatmap_rgb, axis=0))
        # print(f"Predicted label: {np.argmax(prediction)}")


        # # ✅ Kirim heatmap ke CNN (contoh)
        # prediction = self.model.predict(np.expand_dims(heatmap_rgb, axis=0))  # shape: (1, 64, 64, 3)
        # label = np.argmax(prediction)

        # print(f"📢 Predicted label: {label} (confidence: {np.max(prediction):.2f})")
    


        # # Gabungkan semua titik dari 30 frame
        # result = np.vstack(all_points)  # shape: (total_points, 4)

        # # Simpan (opsional)
        # start_ts = int(frameList[0]['timestamp'])
        # np.save(f"output/window_{start_ts}.npy", result)
        # print(f"✅ Saved window_{start_ts}.npy with shape {result.shape}")

    def extract_features(self, frameJSON):
        pc = frameJSON['frameData']['pointCloud']  # shape: (N, 7)
        timestamp = frameJSON['timestamp']

        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        doppler = pc[:, 3]
        snr = pc[:, 4]

        # Hitung range
        range_ = np.sqrt(x**2 + y**2 + z**2)

        # Buat timestamp array (1 timestamp per point)
        timestamp_arr = np.full(range_.shape, timestamp)

        # 🔧 Normalisasi SNR secara langsung di array (tanpa .csv)
        snr_min = 4.68
        snr_max = 2621.36
        snr = np.clip(snr, snr_min, snr_max)
        snr = np.log1p(snr)  # log(1 + snr)

        # Gabungkan jadi (N, 4): timestamp, range, doppler, snr
        return np.stack([timestamp_arr, range_, doppler, snr], axis=1)
    
    def generate_rgb_heatmap_tensor(self, df_subset):
        # Bin definition
        num_x, num_y, num_t = 100, 100, 100
        x_bins = np.linspace(-7, 7, num_x)
        y_bins = np.linspace(0, 5.385, num_y)
        t_bins = np.linspace(df_subset["timestamp"].min(), df_subset["timestamp"].max(), num_t)

        def resize(img):
            img_pil = Image.fromarray(img.astype(np.uint8))
            img_resized = img_pil.resize((64, 64), Image.LANCZOS)
            return np.array(img_resized)

        def save_heatmap(x, y, bins_x, bins_y):
            heatmap, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y], weights=df_subset["SNR"])
            heatmap_counts, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y])
            heatmap_counts[heatmap_counts == 0] = 1  # Hindari divide by zero
            heatmap /= heatmap_counts
            return heatmap.T

        # Buat heatmap DR, DT, RT
        dr = resize(save_heatmap(df_subset["doppler"], df_subset["Range"], x_bins, y_bins))
        dt = resize(save_heatmap(df_subset["timestamp"], df_subset["doppler"], t_bins, x_bins))
        rt = resize(save_heatmap(df_subset["timestamp"], df_subset["Range"], t_bins, y_bins))
        # print("DR min/max:", dr.min(), dr.max())
        # print("DT min/max:", dt.min(), dt.max())
        # print("RT min/max:", rt.min(), rt.max())
        dr = (dr - dr.min()) / (dr.max() - dr.min() + 1e-8)
        dt = (dt - dt.min()) / (dt.max() - dt.min() + 1e-8)
        rt = (rt - rt.min()) / (rt.max() - rt.min() + 1e-8)
        # print("Range range:", df_subset["Range"].min(), df_subset["Range"].max())
        # print("Doppler range:", df_subset["doppler"].min(), df_subset["doppler"].max())
        # Stack jadi RGB dan normalisasi
        heatmap_rgb = np.stack([dr, dt, rt], axis=-1).astype(np.float32)
        heatmap_rgb = (heatmap_rgb - heatmap_rgb.min()) / (heatmap_rgb.max() - heatmap_rgb.min() + 1e-8)
        return dr, dt, rt, heatmap_rgb
        # return heatmap_rgb

    def init_prediction_logger(self):
        import csv, os
        from datetime import datetime
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pred_log_path = f"logs/prediction_log_{timestamp}.csv"
        self.pred_log_file = open(self.pred_log_path, mode="w", newline="")
        self.pred_csv_writer = csv.writer(self.pred_log_file)

        # Tulis header
        self.pred_csv_writer.writerow(['timestamp'] + self.class_names)


    # def prediction_thread_func(self):
    #     while True:
    #         heatmap_rgb, doppler_vals = self.queue.get()  # 🔥 Terima heatmap dan doppler
    #         if heatmap_rgb is None:
    #             break

    #         # 🔥 Start timer
    #         input_tensor = np.expand_dims(heatmap_rgb, axis=0)
    #         start_time = time.time()
    #         pred = self.model.predict(input_tensor, verbose=0)
    #         end_time = time.time()
    #         inference_time = end_time - start_time

    #         # 🔥 Prediksi awal
    #         label_idx = np.argmax(pred)
    #         self.pred_queue.append(label_idx)

    #         # 🔥 Majority voting
    #         # counts = Counter(self.pred_queue)
    #         # majority_label_idx = counts.most_common(1)[0][0]

    #         # label_name = self.class_names[majority_label_idx]
    #         # confidence = pred[0][majority_label_idx]
            
    #         # 🔥 Ambil hasil prediksi
    #         label_idx = np.argmax(pred)
    #         label_name = self.class_names[label_idx]
    #         confidence = pred[0][label_idx]
    #         if hasattr(self, 'prediction_log_buffer'):
    #             now = time.time()
    #             row = [now] + [float(prob) for prob in pred[0]]
    #             self.prediction_log_buffer.append(row)
    #         # label_idx = np.argmax(pred)
    #         # confidence = np.max(pred)
    #         # label_idx, label_name, confidence = self.update_prediction(label_idx, confidence, doppler_vals)


    #         # print(f"📢 Majority Voting Aktivitas: {label_name} (label={majority_label_idx}, conf={confidence:.2f}, infer={inference_time:.4f}s)")

    #         # ✅ Update label di GUI
    #         if hasattr(self, 'guiWindow'):
    #             self.guiWindow.predictionLabel.setText(
    #                 f"Aktivitas: {label_name} ({confidence * 100:.1f}%) | {inference_time:.4f}s"
    #             )
    # def prediction_thread_func(self):
    #     # Inisialisasi jika belum ada
    #     if not hasattr(self, 'last_candidate_label'):
    #         self.last_candidate_label = None
    #         self.candidate_count = 0
    #         self.current_label_idx = None

    #     # Threshold berdasarkan transisi (jumlah berturut-turut dibutuhkan)
    #     self.transition_thresholds = {
    #         ("duduk", "jalan"): 15,
    #         ("jatuh", "jalan"): 15,
    #         ("berdiri", "jalan"): 10,
    #         ("berdiri", "duduk"): 10,
    #         ("berdiri", "jatuh"): 10,
    #         ("jalan", "berdiri"): 10,
    #         ("duduk", "berdiri"): 10,
    #         ("jatuh", "berdiri"): 10,
    #         # default jika tidak dispesifikkan
    #     }

    #     while True:
    #         heatmap_rgb, doppler_vals = self.queue.get()
    #         if heatmap_rgb is None:
    #             break

    #         input_tensor = np.expand_dims(heatmap_rgb, axis=0)
    #         start_time = time.time()
    #         pred = self.model.predict(input_tensor, verbose=0)
    #         end_time = time.time()
    #         inference_time = end_time - start_time

    #         label_idx = np.argmax(pred)
    #         confidence = pred[0][label_idx]

    #         # Simpan ke history
    #         self.pred_queue.append(label_idx)
    #         self.conf_queue.append(confidence)

    #         label_name_candidate = self.class_names[label_idx].lower()
    #         prev_label_name = self.class_names[self.current_label_idx].lower() if self.current_label_idx is not None else None

    #         # Ambil threshold jumlah kemunculan berturut-turut berdasarkan transisi
    #         transition_key = (prev_label_name, label_name_candidate)
    #         threshold = self.transition_thresholds.get(transition_key, 10)

    #         # Tentukan minimum confidence
    #         min_conf = 0.99 if label_name_candidate == "Jalan" else 0.99

    #         # Logika stabilisasi prediksi
    #         if confidence >= min_conf:
    #             if label_idx == self.last_candidate_label:
    #                 self.candidate_count += 1
    #             else:
    #                 self.last_candidate_label = label_idx
    #                 self.candidate_count = 1

    #             if self.candidate_count >= threshold:
    #                 self.current_label_idx = label_idx
    #         else:
    #             self.last_candidate_label = None
    #             self.candidate_count = 0

    #         # Ambil label final
    #         final_label_idx = self.current_label_idx if self.current_label_idx is not None else label_idx
    #         label_name = self.class_names[final_label_idx]
    #         final_confidence = pred[0][final_label_idx]

    #         # Logging jika buffer tersedia
    #         if hasattr(self, 'prediction_log_buffer'):
    #             now = time.time()
    #             row = [now] + [float(prob) for prob in pred[0]]
    #             self.prediction_log_buffer.append(row)

    #         # Tampilkan di GUI jika ada
    #         if hasattr(self, 'guiWindow'):
    #             self.guiWindow.predictionLabel.setText(
    #                 f"Aktivitas: {label_name} ({final_confidence * 100:.1f}%) | {inference_time:.4f}s"
    #             )
    def prediction_thread_func(self):
        # Inisialisasi jika belum ada
        if not hasattr(self, 'last_candidate_label'):
            self.last_candidate_label = None
            self.candidate_count = 0
            self.current_label_idx = None

        while True:
            heatmap_rgb, doppler_vals = self.queue.get()
            if heatmap_rgb is None:
                break

            input_tensor = np.expand_dims(heatmap_rgb, axis=0)
            start_time = time.time()
            pred = self.model.predict(input_tensor, verbose=0)
            end_time = time.time()
            inference_time = end_time - start_time

            label_idx = np.argmax(pred)
            confidence = pred[0][label_idx]

            label_name_candidate = self.class_names[label_idx].lower()

            # Tentukan threshold kemunculan berdasarkan label
            threshold = 10 if label_name_candidate == "jalan" else 5

            # Cek confidence >= 95% dan label sama berturut-turut
            if confidence >= 0.95:
                if label_idx == self.last_candidate_label:
                    self.candidate_count += 1
                else:
                    self.last_candidate_label = label_idx
                    self.candidate_count = 1

                if self.candidate_count >= threshold:
                    self.current_label_idx = label_idx
            else:
                self.last_candidate_label = None
                self.candidate_count = 0

            # Ambil label final
            final_label_idx = self.current_label_idx if self.current_label_idx is not None else label_idx
            label_name = self.class_names[final_label_idx]
            final_confidence = pred[0][final_label_idx]

            # Logging jika buffer tersedia
            if hasattr(self, 'prediction_log_buffer'):
                now = time.time()
                row = [now] + [float(prob) for prob in pred[0]]
                self.prediction_log_buffer.append(row)

            # Tampilkan hasil di GUI jika ada
            if hasattr(self, 'guiWindow'):
                self.guiWindow.predictionLabel.setText(
                    f"Aktivitas: {label_name} ({final_confidence * 100:.1f}%) | {inference_time:.4f}s"
                )

    def update_prediction(self, label_idx, confidence, doppler_values):
            if self.last_label_name == "Jatuh":
                mean_doppler = np.mean(np.abs(doppler_values))
                if mean_doppler > self.doppler_change_threshold:
                    # Aktivitas baru valid, update
                    self.last_label_idx = label_idx
                    self.last_label_name = self.class_names[label_idx]
                    self.last_confidence = confidence
                else:
                    # Tetap di 'jatuh'
                    label_idx = self.last_label_idx
                    label_name = self.last_label_name
                    confidence = self.last_confidence
            else:
                # Bukan jatuh sebelumnya, update biasa
                self.last_label_idx = label_idx
                self.last_label_name = self.class_names[label_idx]
                self.last_confidence = confidence

            return self.last_label_idx, self.last_label_name, self.last_confidence
    def save_prediction_log_to_csv(self):
        if not hasattr(self, 'prediction_log_buffer'):
            return
        import csv
        with open(self.pred_log_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp'] + self.class_names)
            writer.writerows(self.prediction_log_buffer)

    def stop(self):
        if hasattr(self, 'prediction_log_buffer'):
            self.save_prediction_log_to_csv()  # 🔥 Simpan buffer ke CSV

        if hasattr(self, 'pred_log_file'):
            self.pred_log_file.close()

        self.terminate()



class sendCommandThread(QThread):
    done = Signal()

    def __init__(self, uParser, command):
        QThread.__init__(self)
        self.parser = uParser
        self.command = command

    def run(self):
        self.parser.sendLine(self.command)
        self.done.emit()


class updateQTTargetThread3D(QThread):
    done = Signal()

    def __init__(self, pointCloud, targets, scatter, pcplot, numTargets, ellipsoids, coords, colorGradient=None, classifierOut=[], zRange=[-3, 3], pointColorMode="", drawTracks=True, trackColorMap=None, pointBounds={'enabled': False}):
        QThread.__init__(self)
        self.pointCloud = pointCloud
        self.targets = targets
        self.scatter = scatter
        self.pcplot = pcplot
        self.colorArray = ('r', 'g', 'b', 'w')
        self.numTargets = numTargets
        self.ellipsoids = ellipsoids
        self.coordStr = coords
        self.classifierOut = classifierOut
        self.zRange = zRange
        self.colorGradient = colorGradient
        self.pointColorMode = pointColorMode
        self.drawTracks = drawTracks
        self.trackColorMap = trackColorMap
        self.pointBounds = pointBounds
        # This ignores divide by 0 errors when calculating the log2
        np.seterr(divide='ignore')

    def drawTrack(self, track, trackColor):
        # Get necessary track data
        tid = int(track[0])
        x = track[1]
        y = track[2]
        z = track[3]

        track = self.ellipsoids[tid]
        mesh = getBoxLinesCoords(x, y, z)
        track.setData(pos=mesh, color=trackColor, width=2,
                      antialias=True, mode='lines')
        track.setVisible(True)

    # Return transparent color if pointBounds is enabled and point is outside pointBounds
    # Otherwise, color the point depending on which color mode we are in
    def getPointColors(self, i):
        if (self.pointBounds['enabled']):
            xyz_coords = self.pointCloud[i, 0:3]
            if (xyz_coords[0] < self.pointBounds['minX']
                        or xyz_coords[0] > self.pointBounds['maxX']
                        or xyz_coords[1] < self.pointBounds['minY']
                        or xyz_coords[1] > self.pointBounds['maxY']
                        or xyz_coords[2] < self.pointBounds['minZ']
                        or xyz_coords[2] > self.pointBounds['maxZ']
                    ) :
                return pg.glColor((0, 0, 0, 0))

        # Color the points by their SNR
        if (self.pointColorMode == COLOR_MODE_SNR):
            snr = self.pointCloud[i, 4]
            # SNR value is out of expected bounds, make it white
            if (snr < SNR_EXPECTED_MIN) or (snr > SNR_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((snr-SNR_EXPECTED_MIN)/SNR_EXPECTED_RANGE))

        # Color the points by their Height
        elif (self.pointColorMode == COLOR_MODE_HEIGHT):
            zs = self.pointCloud[i, 2]

            # Points outside expected z range, make it white
            if (zs < self.zRange[0]) or (zs > self.zRange[1]):
                return pg.glColor('w')
            else:
                colorRange = self.zRange[1]+abs(self.zRange[0])
                zs = self.zRange[1] - zs
                return pg.glColor(self.colorGradient.getColor(abs(zs/colorRange)))

        # Color Points by their doppler
        elif (self.pointColorMode == COLOR_MODE_DOPPLER):
            doppler = self.pointCloud[i, 3]
            # Doppler value is out of expected bounds, make it white
            if (doppler < DOPPLER_EXPECTED_MIN) or (doppler > DOPPLER_EXPECTED_MAX):
                return pg.glColor('w')
            else:
                return pg.glColor(self.colorGradient.getColor((doppler-DOPPLER_EXPECTED_MIN)/DOPPLER_EXPECTED_RANGE))

        # Color the points by their associate track
        elif (self.pointColorMode == COLOR_MODE_TRACK):
            trackIndex = int(self.pointCloud[i, 6])
            # trackIndex of 253, 254, or 255 indicates a point isn't associated to a track, so check for those magic numbers here
            if (trackIndex == TRACK_INDEX_WEAK_SNR or trackIndex == TRACK_INDEX_BOUNDS or trackIndex == TRACK_INDEX_NOISE):
                return pg.glColor('w')
            else:
                # Catch any errors that may occur if track or point index go out of bounds
                try:
                    return self.trackColorMap[trackIndex]
                except Exception as e:
                    log.error(e)
                    return pg.glColor('w')

        # Unknown Color Option, make all points green
        else:
            return pg.glColor('g')
    #thread 1
    def run(self):

        # if self.pointCloud is None or len(self.pointCloud) == 0:
        #     print("Point Cloud is empty or None.")
        # else:
        #     print("Point Cloud Shape:", self.pointCloud.shape)

        # Clear all previous targets
        for e in self.ellipsoids:
            if (e.visible()):
                e.hide()
        try:
            # Create a list of just X, Y, Z values to be plotted
            if (self.pointCloud is not None):
                toPlot = self.pointCloud[:, 0:3]
                # print("Data for Visualization:", toPlot)

                # Determine the size of each point based on its SNR
                with np.errstate(divide='ignore'):
                    size = np.log2(self.pointCloud[:, 4])

                # Each color is an array of 4 values, so we need an numPoints*4 size 2d array to hold these values
                pointColors = np.zeros((self.pointCloud.shape[0], 4))

                # Set the color of each point
                for i in range(self.pointCloud.shape[0]):
                    pointColors[i] = self.getPointColors(i)

                # Plot the points
                self.scatter.setData(pos=toPlot, color=pointColors, size=size)
                # Debugging
                # print("Pos Data for Visualization:", toPlot)
                # print("Color Data for Visualization:", pointColors)
                # print("Size Data for Visualization:", size)

                # Make the points visible
                self.scatter.setVisible(True)
            else:
                # Make the points invisible if none are detected.
                self.scatter.setVisible(False)
        except Exception as e:
            log.error(
                "Unable to draw point cloud, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")

        # Graph the targets
        try:
            if (self.drawTracks):
                if (self.targets is not None):
                    for track in self.targets:
                        trackID = int(track[0])
                        trackColor = self.trackColorMap[trackID]
                        self.drawTrack(track, trackColor)
        except:
            log.error(
                "Unable to draw all tracks, ignoring and continuing execution...")
            print("Unable to draw point cloud, ignoring and continuing execution...")
            print(f"Error in point cloud visualization: {e}")
        self.done.emit()

    def stop(self):
        self.terminate()
    
