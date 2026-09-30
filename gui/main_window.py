import cv2
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QFileDialog,
    QComboBox,
    QLineEdit,
)

from utils.config import (
    get_cpu_name,
    get_cuda_info,
    detect_cameras,
)

from vision.video_processor import VideoProcessor


class VideoLabel(QLabel):

    VIDEO_WIDTH = 640
    VIDEO_HEIGHT = 360

    def __init__(self, parent=None):

        super().__init__(parent)

        self.setFixedSize(
            self.VIDEO_WIDTH,
            self.VIDEO_HEIGHT
        )

        self.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.setStyleSheet(
            """
            QLabel {
                background-color: #111111;
                border: 1px solid #444444;
            }
            """
        )

        self.setScaledContents(False)

    def display_frame(self, frame):

        if frame is None:
            return

        height, width = frame.shape[:2]

        if width <= 0 or height <= 0:
            return

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        bytes_per_line = width * 3

        image = QImage(
            rgb_frame.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )

        pixmap = QPixmap.fromImage(
            image
        )

        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )

        self.setPixmap(
            scaled
        )

    def clear_frame(self):

        self.clear()


class MainWindow(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "Crowd Movement Analysis"
        )

        self.resize(
            1400,
            1200
        )

        self.setMinimumSize(
            1200,
            900
        )

        # ==========================================
        # VIDEO STATE
        # ==========================================

        self.cap = None

        self.camera_active = False
        self.video_active = False
        self.stream_active = False

        self.selected_camera_index = None

        # ==========================================
        # PROCESSOR
        # ==========================================

        self.processor = VideoProcessor("GPU")

        # ==========================================
        # FPS / TIMING
        # ==========================================

        self.processing_frames = 0
        self.fps_start_time = None

        # ==========================================
        # TIMER
        # ==========================================

        self.timer = QTimer()

        self.timer.timeout.connect(
            self.update_frame
        )

        # ==========================================
        # HARDWARE INFORMATION
        # ==========================================

        self.cpu_name = get_cpu_name()
        self.cuda_info = get_cuda_info()

        # ==========================================
        # VIDEO DISPLAY WIDGETS
        # ==========================================

        self.input_video = VideoLabel()
        self.output_video = VideoLabel()

        # ==========================================
        # BUILD GUI
        # ==========================================

        self.build_ui()

        # ==========================================
        # CAMERA DETECTION
        # ==========================================

        self.refresh_cameras()

        # ==========================================
        # DEFAULT DEVICE
        # ==========================================

        self.device_changed(
            self.device_selector.currentText()
        )

    # ==================================================
    # BUILD UI
    # ==================================================

    def build_ui(self):

        central_widget = QWidget()

        self.setCentralWidget(
            central_widget
        )

        main_layout = QVBoxLayout(
            central_widget
        )

        # ==================================================
        # TITLE
        # ==================================================

        title = QLabel(
            "CROWD MOVEMENT ANALYSIS"
        )

        title.setAlignment(
            Qt.AlignCenter
        )

        title.setStyleSheet("""
            QLabel {
                font-size: 26px;
                font-weight: bold;
                padding: 12px;
            }
        """)

        main_layout.addWidget(title)

        # ==================================================
        # VIDEO PANELS
        # ==================================================

        video_layout = QHBoxLayout()

        input_group = QGroupBox(
            "INPUT VIDEO"
        )

        input_layout = QVBoxLayout()

        input_layout.addWidget(
            self.input_video,
            alignment=Qt.AlignmentFlag.AlignCenter
        )

        input_group.setLayout(
            input_layout
        )

        output_group = QGroupBox(
            "PROCESSED OUTPUT"
        )

        output_layout = QVBoxLayout()

        output_layout.addWidget(
            self.output_video,
            alignment=Qt.AlignmentFlag.AlignCenter
        )

        output_group.setLayout(
            output_layout
        )

        video_layout.addWidget(
            input_group
        )

        video_layout.addWidget(
            output_group
        )

        main_layout.addLayout(
            video_layout
        )

        # ==================================================
        # INPUT SOURCE
        # ==================================================

        source_group = QGroupBox(
            "INPUT SOURCE"
        )

        source_layout = QGridLayout()

        # ------------------------------------------
        # Source selector
        # ------------------------------------------

        source_layout.addWidget(
            QLabel("Source Type:"),
            0,
            0
        )

        self.source_selector = QComboBox()

        self.source_selector.addItems([
            "USB Camera",
            "Recorded Video",
            "Network Stream"
        ])

        self.source_selector.currentTextChanged.connect(
            self.source_changed
        )

        source_layout.addWidget(
            self.source_selector,
            0,
            1,
            1,
            2
        )

        # ------------------------------------------
        # Camera selector
        # ------------------------------------------

        self.camera_label = QLabel(
            "Camera:"
        )

        source_layout.addWidget(
            self.camera_label,
            1,
            0
        )

        self.camera_selector = QComboBox()

        source_layout.addWidget(
            self.camera_selector,
            1,
            1
        )

        self.refresh_camera_button = QPushButton(
            "Refresh Cameras"
        )

        self.refresh_camera_button.clicked.connect(
            self.refresh_cameras
        )

        source_layout.addWidget(
            self.refresh_camera_button,
            1,
            2
        )

        # ------------------------------------------
        # Stream URL
        # ------------------------------------------

        self.stream_label = QLabel(
            "Stream URL:"
        )

        source_layout.addWidget(
            self.stream_label,
            2,
            0
        )

        self.stream_url = QLineEdit()

        self.stream_url.setPlaceholderText(
            "rtsp://192.168.1.100:554/stream"
        )

        source_layout.addWidget(
            self.stream_url,
            2,
            1
        )

        self.connect_stream_button = QPushButton(
            "Connect Stream"
        )

        self.connect_stream_button.clicked.connect(
            self.connect_stream
        )

        source_layout.addWidget(
            self.connect_stream_button,
            2,
            2
        )

        # ------------------------------------------
        # Open video
        # ------------------------------------------

        self.open_video_button = QPushButton(
            "Open Video File"
        )

        self.open_video_button.clicked.connect(
            self.open_video
        )

        source_layout.addWidget(
            self.open_video_button,
            3,
            0
        )

        # ------------------------------------------
        # Start camera
        # ------------------------------------------

        self.live_camera_button = QPushButton(
            "Connect Camera"
        )

        self.live_camera_button.clicked.connect(
            self.start_camera
        )

        source_layout.addWidget(
            self.live_camera_button,
            3,
            1
        )

        source_group.setLayout(
            source_layout
        )

        main_layout.addWidget(
            source_group
        )

        # ==================================================
        # CROWD PARAMETERS
        # ==================================================

        parameters_group = QGroupBox(
            "CROWD PARAMETERS"
        )

        parameters_layout = QGridLayout()

        # ------------------------------------------
        # People Count
        # ------------------------------------------

        self.people_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("People Count:"),
            0,
            0
        )

        parameters_layout.addWidget(
            self.people_value,
            0,
            1
        )

        # ------------------------------------------
        # Density
        # ------------------------------------------

        self.density_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Density:"),
            0,
            2
        )

        parameters_layout.addWidget(
            self.density_value,
            0,
            3
        )

        # ------------------------------------------
        # Direction
        # ------------------------------------------

        self.direction_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Direction:"),
            1,
            0
        )

        parameters_layout.addWidget(
            self.direction_value,
            1,
            1
        )

        # ------------------------------------------
        # Average Motion
        # ------------------------------------------

        self.speed_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Average Motion:"),
            1,
            2
        )

        parameters_layout.addWidget(
            self.speed_value,
            1,
            3
        )

        # ------------------------------------------
        # Motion Intensity
        # ------------------------------------------

        self.motion_intensity_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Motion Intensity:"),
            2,
            0
        )

        parameters_layout.addWidget(
            self.motion_intensity_value,
            2,
            1
        )

        # ------------------------------------------
        # Flow Consistency
        # ------------------------------------------

        self.flow_consistency_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Flow Consistency:"),
            2,
            2
        )

        parameters_layout.addWidget(
            self.flow_consistency_value,
            2,
            3
        )

        # ------------------------------------------
        # Occupancy
        # ------------------------------------------

        self.occupancy_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Occupancy:"),
            3,
            0
        )

        parameters_layout.addWidget(
            self.occupancy_value,
            3,
            1
        )

        # ------------------------------------------
        # Flow
        # ------------------------------------------

        self.flow_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Flow:"),
            3,
            2
        )

        parameters_layout.addWidget(
            self.flow_value,
            3,
            3
        )

        # ------------------------------------------
        # Congestion
        # ------------------------------------------

        self.congestion_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Congestion:"),
            4,
            0
        )

        parameters_layout.addWidget(
            self.congestion_value,
            4,
            1
        )

        # ------------------------------------------
        # Congested Zones
        # ------------------------------------------

        self.congested_zones_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Congested Zones:"),
            4,
            2
        )

        parameters_layout.addWidget(
            self.congested_zones_value,
            4,
            3
        )

        # ------------------------------------------
        # Congestion Share
        # ------------------------------------------

        self.congestion_share_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("Congestion Share:"),
            5,
            0
        )

        parameters_layout.addWidget(
            self.congestion_share_value,
            5,
            1
        )

        # ------------------------------------------
        # People in Congested Zones
        # ------------------------------------------
        self.people_congestion_value = QLabel("--")

        parameters_layout.addWidget(
            QLabel("People in Congested Zones:"),
            5,
            2
        )

        parameters_layout.addWidget(
            self.people_congestion_value,
            5,
            3
        )

        # ------------------------------------------
        # SET CROWD PARAMETERS LAYOUT
        # ------------------------------------------

        parameters_group.setLayout(
            parameters_layout
        )

        main_layout.addWidget(
            parameters_group
        )

        # ==================================================
        # PROCESSING
        # ==================================================

        processing_group = QGroupBox(
            "PROCESSING"
        )


        processing_layout = QGridLayout()

        # ------------------------------------------
        # Processing device
        # ------------------------------------------

        processing_layout.addWidget(
            QLabel("Processing Device:"),
            0,
            0
        )

        self.device_selector = QComboBox()

        self.device_selector.addItems([
            "CPU",
            "GPU"
        ])

        self.device_selector.setCurrentText(
            "GPU"
        )

        self.device_selector.currentTextChanged.connect(
            self.device_changed
        )

        processing_layout.addWidget(
            self.device_selector,
            0,
            1
        )

        # ------------------------------------------
        # Device
        # ------------------------------------------

        processing_layout.addWidget(
            QLabel("Device:"),
            1,
            0
        )

        self.device_value = QLabel("--")

        self.device_value.setWordWrap(
            True
        )

        processing_layout.addWidget(
            self.device_value,
            1,
            1
        )

        # ------------------------------------------
        # CUDA
        # ------------------------------------------

        processing_layout.addWidget(
            QLabel("CUDA:"),
            1,
            2
        )

        self.cuda_value = QLabel("--")

        processing_layout.addWidget(
            self.cuda_value,
            1,
            3
        )

        # ------------------------------------------
        # FPS
        # ------------------------------------------

        processing_layout.addWidget(
            QLabel("Processing FPS:"),
            2,
            0
        )

        self.fps_value = QLabel("--")

        processing_layout.addWidget(
            self.fps_value,
            2,
            1
        )

        # ------------------------------------------
        # Frame time
        # ------------------------------------------

        processing_layout.addWidget(
            QLabel("Frame Time:"),
            2,
            2
        )

        self.frame_time_value = QLabel("--")

        processing_layout.addWidget(
            self.frame_time_value,
            2,
            3
        )

        # ------------------------------------------
        # OpenCV CUDA
        # ------------------------------------------

        processing_layout.addWidget(
            QLabel("OpenCV CUDA:"),
            3,
            0
        )

        self.opencv_cuda_value = QLabel("--")

        processing_layout.addWidget(
            self.opencv_cuda_value,
            3,
            1
        )

        # ------------------------------------------
        # Resolution
        # ------------------------------------------

        processing_layout.addWidget(
            QLabel("Resolution:"),
            3,
            2
        )

        self.resolution_value = QLabel("--")

        processing_layout.addWidget(
            self.resolution_value,
            3,
            3
        )

        processing_group.setLayout(
            processing_layout
        )

        main_layout.addWidget(
            processing_group
        )

        # ==================================================
        # CONTROL BUTTONS
        # ==================================================

        control_layout = QHBoxLayout()

        self.start_button = QPushButton(
            "Start"
        )

        self.pause_button = QPushButton(
            "Pause"
        )

        self.stop_button = QPushButton(
            "Stop"
        )

        self.start_button.clicked.connect(
            self.start_processing
        )

        self.pause_button.clicked.connect(
            self.pause_processing
        )

        self.stop_button.clicked.connect(
            self.stop_processing
        )

        control_layout.addWidget(
            self.start_button
        )

        control_layout.addWidget(
            self.pause_button
        )

        control_layout.addWidget(
            self.stop_button
        )

        main_layout.addLayout(
            control_layout
        )

        # ==================================================
        # STATUS
        # ==================================================

        self.status_label = QLabel(
            "Status: Waiting for input source"
        )

        self.status_label.setAlignment(
            Qt.AlignCenter
        )

        self.status_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                font-weight: bold;
            }
        """)

        main_layout.addWidget(
            self.status_label
        )

        # Set initial source UI
        self.source_changed(
            self.source_selector.currentText()
        )

    # ==================================================
    # SOURCE UI
    # ==================================================

    def source_changed(self, source):

        if source == "USB Camera":

            # Camera controls
            self.camera_label.show()
            self.camera_selector.show()
            self.refresh_camera_button.show()
            self.live_camera_button.show()

            # Hide recorded video controls
            self.open_video_button.hide()

            # Hide network stream controls
            self.stream_label.hide()
            self.stream_url.hide()
            self.connect_stream_button.hide()

        elif source == "Recorded Video":

            # Hide camera controls
            self.camera_label.hide()
            self.camera_selector.hide()
            self.refresh_camera_button.hide()
            self.live_camera_button.hide()

            # Show recorded video control
            self.open_video_button.show()

            # Hide network stream controls
            self.stream_label.hide()
            self.stream_url.hide()
            self.connect_stream_button.hide()

        elif source == "Network Stream":

            # Hide camera controls
            self.camera_label.hide()
            self.camera_selector.hide()
            self.refresh_camera_button.hide()
            self.live_camera_button.hide()

            # Hide recorded video control
            self.open_video_button.hide()

            # Show network stream controls
            self.stream_label.show()
            self.stream_url.show()
            self.connect_stream_button.show()

    # ==================================================
    # CAMERA DETECTION
    # ==================================================

    def refresh_cameras(self):

        current_index = (
            self.camera_selector.currentData()
        )

        self.camera_selector.clear()

        self.status_label.setText(
            "Status: Detecting cameras..."
        )

        cameras = detect_cameras(
            max_cameras=10
        )

        if not cameras:

            self.camera_selector.addItem(
                "No camera detected",
                None
            )

            self.status_label.setText(
                "Status: No camera detected"
            )

            return

        for index, description in cameras:

            self.camera_selector.addItem(
                description,
                index
            )

        # Restore previous camera
        if current_index is not None:

            position = (
                self.camera_selector.findData(
                    current_index
                )
            )

            if position >= 0:

                self.camera_selector.setCurrentIndex(
                    position
                )

        self.status_label.setText(
            f"Status: {len(cameras)} camera(s) detected"
        )

    # ==================================================
    # DEVICE SELECTION
    # ==================================================

    def device_changed(self, device):

        self.processor.set_device(
            device
        )

        if device == "CPU":

            self.device_value.setText(
                self.cpu_name
            )

            self.cuda_value.setText(
                "Not Used"
            )

            self.opencv_cuda_value.setText(
                "Not Used"
            )

            self.status_label.setText(
                "Status: CPU processing selected"
            )

        elif device == "GPU":

            if self.cuda_info["available"]:

                self.device_value.setText(
                    self.cuda_info["name"]
                )

                self.cuda_value.setText(
                    "Available"
                )

                self.opencv_cuda_value.setText(
                    "Available"
                )

                self.status_label.setText(
                    "Status: GPU processing selected"
                )

            else:

                self.device_value.setText(
                    "No CUDA GPU detected"
                )

                self.cuda_value.setText(
                    "Unavailable"
                )

                self.opencv_cuda_value.setText(
                    "Unavailable"
                )

                self.status_label.setText(
                    "Status: CUDA GPU unavailable"
                )

    # ==================================================
    # USB CAMERA
    # ==================================================

    def start_camera(self):

        camera_index = (
            self.camera_selector.currentData()
        )

        if camera_index is None:

            self.status_label.setText(
                "Status: No camera selected"
            )

            return

        self.stop_video()

        self.status_label.setText(
            f"Status: Opening Camera {camera_index}..."
        )

        self.cap = cv2.VideoCapture(
            camera_index,
            cv2.CAP_DSHOW
        )

        if not self.cap.isOpened():

            self.cap.release()
            self.cap = None

            self.status_label.setText(
                "Status: Camera could not be opened"
            )

            return

        self.selected_camera_index = (
            camera_index
        )

        # Request resolution
        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        # Request 30 FPS
        self.cap.set(
            cv2.CAP_PROP_FPS,
            30
        )

        self.processor.reset()

        self.camera_active = True
        self.video_active = False
        self.stream_active = False

        self.update_capture_information()

        self.status_label.setText(
            f"Status: Camera {camera_index} connected"
        )

    # ==================================================
    # RECORDED VIDEO
    # ==================================================

    def open_video(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov)"
        )

        if not file_path:
            return

        self.stop_video()

        self.cap = cv2.VideoCapture(
            file_path
        )

        if not self.cap.isOpened():

            self.cap.release()
            self.cap = None

            self.status_label.setText(
                "Status: Could not open video"
            )

            return

        self.processor.reset()

        self.camera_active = False
        self.video_active = True
        self.stream_active = False

        self.update_capture_information()

        self.status_label.setText(
            "Status: Recorded video loaded"
        )

    # ==================================================
    # NETWORK STREAM
    # ==================================================

    def connect_stream(self):

        url = self.stream_url.text().strip()

        if not url:

            self.status_label.setText(
                "Status: Enter a stream URL"
            )

            return

        self.stop_video()

        self.status_label.setText(
            "Status: Connecting to network stream..."
        )

        # Open RTSP / HTTP / direct video stream
        self.cap = cv2.VideoCapture(
            url
        )

        if not self.cap.isOpened():

            self.cap.release()
            self.cap = None

            self.status_label.setText(
                "Status: Could not connect to stream"
            )

            return

        self.processor.reset()

        self.camera_active = False
        self.video_active = False
        self.stream_active = True

        self.update_capture_information()

        self.status_label.setText(
            "Status: Network stream connected"
        )

    # ==================================================
    # CAPTURE INFORMATION
    # ==================================================

    def update_capture_information(self):

        if self.cap is None:
            return

        width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            fps = 0

        self.resolution_value.setText(
            f"{width} × {height} @ {fps:.1f} FPS"
        )

    # ==================================================
    # START PROCESSING
    # ==================================================

    def start_processing(self):

        if self.cap is None:

            self.status_label.setText(
                "Status: Select an input source first"
            )

            return

        self.processor.reset()

        self.processing_frames = 0

        self.fps_start_time = (
            time.perf_counter()
        )

        self.timer.start(10)

        self.status_label.setText(
            "Status: Processing"
        )

    # ==================================================
    # PAUSE
    # ==================================================

    def pause_processing(self):

        self.timer.stop()

        self.status_label.setText(
            "Status: Paused"
        )

    # ==================================================
    # STOP
    # ==================================================

    def stop_processing(self):

        self.timer.stop()

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        self.camera_active = False
        self.video_active = False
        self.stream_active = False

        self.input_video.clear()

        self.input_video.setText(
            "Camera / Recorded Video / Stream"
        )

        self.output_video.clear()

        self.output_video.setText(
            "Crowd Movement Output"
        )

        # Reset crowd values
        self.people_value.setText("--")
        self.density_value.setText("--")
        self.direction_value.setText("--")
        self.speed_value.setText("--")
        self.motion_intensity_value.setText("--")
        self.flow_consistency_value.setText("--")
        self.occupancy_value.setText("--")
        self.flow_value.setText("--")
        self.congestion_value.setText("--")
        self.congested_zones_value.setText("--")
        self.congestion_share_value.setText("--")
        self.people_congestion_value.setText("--")

        # Reset processing values
        self.fps_value.setText("--")
        self.frame_time_value.setText("--")
        self.resolution_value.setText("--")

        self.status_label.setText(
            "Status: Stopped"
        )

    # ==================================================
    # STOP VIDEO SOURCE
    # ==================================================

    def stop_video(self):

        self.timer.stop()

        if self.cap is not None:

            self.cap.release()
            self.cap = None

        self.camera_active = False
        self.video_active = False
        self.stream_active = False

    # ==================================================
    # FRAME UPDATE
    # ==================================================


    def update_frame(self):

        if self.cap is None:
            return

        start_time = time.perf_counter()

        ret, frame = self.cap.read()

        if not ret:

            if self.video_active:

                self.status_label.setText(
                    "Status: Video finished"
                )

            elif self.stream_active:

                self.status_label.setText(
                    "Status: Stream stopped / disconnected"
                )

            else:

                self.status_label.setText(
                    "Status: Unable to read frame"
                )

            self.timer.stop()

            return

        # ==========================================
        # INPUT VIDEO
        # ==========================================

        self.input_video.display_frame(
            frame
        )

        # ==========================================
        # PROCESS
        # ==========================================

        result = self.processor.process(
            frame
        )

        # ==========================================
        # OUTPUT VIDEO
        # ==========================================

        self.output_video.display_frame(
            result["frame"]
        )

        # ==========================================
        # CROWD PARAMETERS
        # ==========================================

        # ------------------------------------------
        # People Count
        # ------------------------------------------

        self.people_value.setText(
            str(
                result["people_count"]
            )
        )

        # ------------------------------------------
        # Density
        # ------------------------------------------

        self.density_value.setText(
            f'{result["density"]:.2f} people/100k px'
        )

        # ------------------------------------------
        # Direction
        # ------------------------------------------

        self.direction_value.setText(
            result["direction"]
        )

        # ------------------------------------------
        # Average Motion
        # ------------------------------------------

        self.speed_value.setText(
            f'{result["average_motion"]:.2f} px/frame'
        )

        # ------------------------------------------
        # Motion Intensity
        # ------------------------------------------

        self.motion_intensity_value.setText(
            result["motion_intensity"]
        )

        # ------------------------------------------
        # Flow Consistency
        # ------------------------------------------

        self.flow_consistency_value.setText(
            f'{result["flow_consistency"]:.1f}%'
        )

        # ------------------------------------------
        # Occupancy
        # ------------------------------------------

        self.occupancy_value.setText(
            f'{result["occupancy"]:.1f}%'
        )

        # ------------------------------------------
        # People Flow
        # ------------------------------------------

        self.flow_value.setText(
            f'{result["flow"]:.2f} people/s'
        )

        # ------------------------------------------
        # Congestion
        # ------------------------------------------

        self.congestion_value.setText(
            result["congestion"]
        )

        # ------------------------------------------
        # Congested Zones
        # ------------------------------------------

        self.congested_zones_value.setText(
            f'{result["congested_zones"]} / 9'
        )

        # ------------------------------------------
        # Congestion Share
        # ------------------------------------------

        self.congestion_share_value.setText(
            f'{result["congestion_share"]:.1f}%'
        )

        # ------------------------------------------
        # People in Congested Zones
        # ------------------------------------------

        self.people_congestion_value.setText(
            f'{result["people_in_congested_zones"]} / '
            f'{result["people_count"]}'
        )

        # ==========================================
        # TIMING
        # ==========================================

        end_time = time.perf_counter()

        frame_time = (
            end_time - start_time
        )

        if frame_time > 0:

            instant_fps = (
                1.0 / frame_time
            )

            self.frame_time_value.setText(
                f"{frame_time * 1000:.2f} ms"
            )

            self.fps_value.setText(
                f"{instant_fps:.2f} FPS"
            )

        # ==========================================
        # AVERAGE FPS
        # ==========================================

        self.processing_frames += 1

        if self.fps_start_time is None:

            self.fps_start_time = end_time

        elapsed = (
            end_time -
            self.fps_start_time
        )

        if elapsed >= 1.0:

            average_fps = (
                self.processing_frames /
                elapsed
            )

            self.fps_value.setText(
                f"{average_fps:.2f} FPS"
            )

            self.processing_frames = 0

            self.fps_start_time = end_time

    # ==================================================
    # CLOSE
    # ==================================================

    def closeEvent(self, event):

        self.stop_video()

        event.accept()

