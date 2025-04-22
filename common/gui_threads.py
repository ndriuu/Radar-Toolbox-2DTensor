# General Library Imports
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

    def __init__(self, uParser, window_size=30, stride=1,viewer=None):
        QThread.__init__(self)
        self.parser = uParser
        # self.viewer = viewer
        self.model = load_model("andrikun_fixed.h5", compile=False)
        # self.model.summary()  # ⬅️ Panggil setelah model berhasil diload
        self.class_names = ["Berdiri", "Duduk", "Jalan", "Jatuh"]

        # self.x, self.y, self.z = 10, 32, 32
        # self.x_min, self.x_max = -1.5, 1.5
        # self.y_min, self.y_max = 0, 4
        # self.z_min, self.z_max = 0, 2

        # self.x_res = (self.x_max - self.x_min) / self.x
        # self.y_res = (self.y_max - self.y_min) / self.y
        # self.z_res = (self.z_max - self.z_min) / self.z
        self.frameBuffer = deque(maxlen=window_size)
        self.window_size = window_size
        self.stride = stride
        self.counter = 0
        self.timestamp = time.strftime("%m%d%Y%H%M%S")
        self.outputDir = f'./dataset/{self.timestamp}'
        # Ensure the directory is created only once
        os.makedirs(self.outputDir, exist_ok=True)

    def run(self):
        if self.parser.parserType == "SingleCOMPort":
            outputDict = self.parser.readAndParseUartSingleCOMPort()
        else:
            outputDict = self.parser.readAndParseUartDoubleCOMPort()

            frameJSON = {
                'frameData': outputDict,
                'timestamp': time.time() * 1000
            }
        self.fin.emit(outputDict)


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
        # Gabungkan semua titik dari 30 frame
        all_points = []
        # all_points1 = []
        for frame in frameList:
            extracted = self.extract_features(frame)  # hasil: (N, 4)
            all_points.append(extracted)
            # extracted1 = self.extract_features(frame)  # hasil: (N, 4)
            # all_points1.append(extracted1)

        result = np.vstack(all_points)
        # result1 = np.vstack(all_points)
        # x, y, z, doppler, snr = result[:, 0], result[:, 1], result[:, 2], result[:, 3], result[:, 4]
        # voxel = self.voxelize(x, y, z, doppler, snr)

        df_subset = {
            "timestamp": result[:, 0],
            "Range": result[:, 1],
            "doppler": result[:, 2],
            "SNR": result[:, 3],
        }

        # 🔥 Buat RGB tensor
        dr, dt, rt, heatmap_rgb = self.generate_rgb_heatmap_tensor(df_subset)
        input_tensor = np.expand_dims(heatmap_rgb, axis=0)

        # Normalisasi jika model kamu pakai skala 0-1
        # input_tensor = input_tensor.astype(np.float32) / 255.0  # ✅ hanya jika training juga pakai ini

        # Prediksi
        pred = self.model.predict(input_tensor, verbose=0)  # verbose=0 biar gak spam
        label_idx = np.argmax(pred)
        confidence = np.max(pred)
        label_name = self.class_names[label_idx]

        print(f"📢 Prediksi Aktivitas: {label_name} (label={label_idx}, conf={confidence:.2f})")
        if hasattr(self, 'guiWindow'):
            self.guiWindow.predictionLabel.setText(
                f"Aktivitas: {label_name} ({confidence * 100:.1f}%)"
            )
            self.guiWindow.updateHeatmapGUI(dr, dt, rt)

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
    
    # def debug_visualize_heatmap(self, heatmap_rgb, start_ts):
    #     import matplotlib.pyplot as plt
    #     os.makedirs("debug", exist_ok=True)
    #     plt.imsave(f"debug/heatmap_window_{start_ts}.png", heatmap_rgb)
    #     print(f"✅ Heatmap disimpan ke debug/heatmap_window_{start_ts}.png")

    # def extract_features(self, frameJSON):
    #     pc = frameJSON['frameData']['pointCloud']
    #     timestamp = frameJSON['timestamp']
        
    #     x = pc[:, 0]
    #     y = pc[:, 1]
    #     z = pc[:, 2]
    #     doppler = pc[:, 3]
    #     snr = pc[:, 4]

    #     # Normalisasi SNR (log-scale)
    #     snr = np.clip(snr, 4.68, 2621.36)
    #     snr = np.log1p(snr)

    #     return np.stack([x, y, z, doppler, snr], axis=1)

    # def voxelize(self, x, y, z, doppler, snr):
    #     valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    #     x, y, z, doppler, snr = x[valid_mask], y[valid_mask], z[valid_mask], doppler[valid_mask], snr[valid_mask]

    #     if len(x) == 0:
    #         return np.zeros([self.x, self.y, self.z, 3])

    #     pixel = np.zeros([self.x, self.y, self.z, 3])
    #     pixel_counts = np.zeros([self.x, self.y, self.z])

    #     for i in range(len(x)):
    #         try:
    #             x_idx = int((x[i] - self.x_min) / self.x_res)
    #             y_idx = int((y[i] - self.y_min) / self.y_res)
    #             z_idx = int((z[i] - self.z_min) / self.z_res)

    #             if 0 <= x_idx < self.x and 0 <= y_idx < self.y and 0 <= z_idx < self.z:
    #                 pixel[x_idx, y_idx, z_idx, 0] += 1
    #                 pixel[x_idx, y_idx, z_idx, 1] += doppler[i]
    #                 pixel[x_idx, y_idx, z_idx, 2] += snr[i]
    #                 pixel_counts[x_idx, y_idx, z_idx] += 1
    #         except Exception:
    #             continue

    #     nonzero_idx = pixel_counts > 0
    #     pixel[nonzero_idx, 1] /= pixel_counts[nonzero_idx]
    #     pixel[nonzero_idx, 2] /= pixel_counts[nonzero_idx]

    #     return pixel

    def stop(self):
        self.terminate()

    def stop(self):
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
    
