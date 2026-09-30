import cv2
import numpy as np
import time

from collections import OrderedDict

from ultralytics import YOLO


class VideoProcessor:

    def __init__(self, device="GPU"):

        self.device = device

        # ==========================================
        # YOLO
        # Supporting module:
        # Used to detect people
        # ==========================================

        self.model = YOLO("yolo26n.pt")

        self.person_class = 0

        # Lower confidence helps detect smaller/farther people
        self.confidence_threshold = 0.30

        # YOLO input resolution
        self.yolo_imgsz = 1280
        

        # Run YOLO every N frames
        self.detection_interval = 2
        self.frame_counter = 0

        self.last_detections = []

        # ==========================================
        # PERSON TRACKING
        # ==========================================

        self.next_person_id = 1

        self.tracks = OrderedDict()

        self.max_trail_length = 30

        self.max_tracking_distance = 100

        # ==========================================
        # OPTICAL FLOW
        # ==========================================

        self.prev_gray = None

        self.flow_gpu = None
        self.cuda_flow_available = False

        self.previous_gpu = None
        self.current_gpu = None

        self.current_flow = None
        self.flow_magnitude = None
        self.flow_angle = None

        self._initialize_optical_flow()

        # ==========================================
        # CROWD RESULTS
        # ==========================================

        self.people_count = 0

        self.direction = "--"

        self.average_motion = 0.0

        self.motion_intensity = "--"

        self.flow_consistency = 0.0

        # ==========================================
        # DENSITY
        # ==========================================

        self.density = 0.0

        self.density_unit = "people/100k px"

        # ==========================================
        # OCCUPANCY
        # ==========================================

        self.occupancy = 0.0

        # ==========================================
        # PEOPLE FLOW
        # ==========================================

        self.people_flow = 0.0

        self.flow_crossings = 0

        self.flow_time_window = 5.0

        self.flow_crossing_history = []

        # ==========================================
        # COUNTING LINE
        # ==========================================

        self.counting_line_ratio = 0.50

        # ==========================================
        # CONGESTION
        # ==========================================

        self.congestion = "LOW"

        self.congested_zones = 0

        # 3 × 3 internal analysis grid
        self.zone_rows = 3
        self.zone_cols = 3

        self.total_zones = (
            self.zone_rows *
            self.zone_cols
        )

        # Zone congestion thresholds
        self.zone_density_threshold = 2
        self.zone_occupancy_threshold = 6.0
        self.zone_motion_threshold = 1.5

        # Additional congestion statistics
        self.congestion_share = 0.0
        self.people_in_congested_zones = 0
        self.people_congestion_share = 0.0

        # ==========================================
        # DIRECTION SMOOTHING
        # ==========================================

        self.direction_history = []

        self.direction_history_length = 8

        self.current_direction = "--"

        self.stationary_threshold = 0.8

        # ==========================================
        # PERSON FLOW
        # ==========================================

        self.person_flow_threshold = 0.8
        self.average_motion_threshold = 0.1

        self.min_person_flow_vectors = 10

        self.person_direction_history_length = 6

        # ==========================================
        # PERFORMANCE
        # ==========================================

        self.last_yolo_time = 0.0

        self.last_flow_time = 0.0

        # ==========================================
        # DEVICE
        # ==========================================

        self.set_device(device)

    # ==================================================
    # DEVICE
    # ==================================================

    def set_device(self, device):

        self.device = device

        self.reset()

        if device == "GPU":

            try:

                import torch

                if torch.cuda.is_available():

                    self.model.to("cuda:0")

                    print(
                        "YOLO device: CUDA"
                    )

                    print(
                        "GPU:",
                        torch.cuda.get_device_name(0)
                    )

                else:

                    print(
                        "PyTorch CUDA unavailable. "
                        "Using CPU."
                    )

                    self.device = "CPU"

                    self.model.to("cpu")

            except Exception as e:

                print(
                    "GPU initialization failed:",
                    e
                )

                self.device = "CPU"

                try:

                    self.model.to("cpu")

                except Exception:

                    pass

        else:

            try:

                self.model.to("cpu")

                print(
                    "YOLO device: CPU"
                )

            except Exception:

                pass

    # ==================================================
    # OPTICAL FLOW INITIALIZATION
    # ==================================================

    def _initialize_optical_flow(self):

        self.flow_gpu = None

        self.cuda_flow_available = False

        try:

            if cv2.cuda.getCudaEnabledDeviceCount() <= 0:

                print(
                    "OpenCV CUDA device unavailable."
                )

                return

            if hasattr(
                cv2.cuda,
                "FarnebackOpticalFlow_create"
            ):

                self.flow_gpu = (
                    cv2.cuda.FarnebackOpticalFlow_create(
                        5,
                        0.5,
                        False,
                        15,
                        3,
                        5,
                        1.2,
                        0
                    )
                )

                self.cuda_flow_available = True

            elif hasattr(
                cv2.cuda,
                "cuda_FarnebackOpticalFlow"
            ):

                self.flow_gpu = (
                    cv2.cuda.cuda_FarnebackOpticalFlow.create(
                        5,
                        0.5,
                        False,
                        15,
                        3,
                        5,
                        1.2,
                        0
                    )
                )

                self.cuda_flow_available = True

            if self.cuda_flow_available:

                print(
                    "OpenCV CUDA optical flow: ENABLED"
                )

        except Exception as e:

            print(
                "CUDA optical flow initialization failed:",
                e
            )

            self.flow_gpu = None

            self.cuda_flow_available = False

    # ==================================================
    # PERSON DETECTION
    # ==================================================

    def detect_people(self, frame):

        inference_device = (
            "cuda:0"
            if self.device == "GPU"
            else "cpu"
        )

        start = time.perf_counter()

        try:

            results = self.model.predict(
                source=frame,
                device=inference_device,
                classes=[self.person_class],
                conf=self.confidence_threshold,
                imgsz=self.yolo_imgsz,
                verbose=False
            )

        except Exception as e:

            if self.device == "GPU":

                print(
                    "YOLO CUDA inference failed:",
                    e
                )

                print(
                    "Falling back to CPU."
                )

                self.device = "CPU"

                try:

                    self.model.to("cpu")

                    results = self.model.predict(
                        source=frame,
                        device="cpu",
                        classes=[self.person_class],
                        conf=self.confidence_threshold,
                        imgsz=self.yolo_imgsz,
                        verbose=False
                    )

                except Exception:

                    return []

            else:

                return []

        self.last_yolo_time = (
            time.perf_counter() - start
        ) * 1000.0

        if not results:

            return []

        result = results[0]

        if result.boxes is None:

            return []

        detections = []

        boxes = result.boxes

        for i in range(len(boxes)):

            xyxy = (
                boxes.xyxy[i]
                .detach()
                .cpu()
                .numpy()
            )

            confidence = float(
                boxes.conf[i]
                .detach()
                .cpu()
                .item()
            )

            x1, y1, x2, y2 = map(
                int,
                xyxy
            )

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )

            detections.append(
                {
                    "box": (
                        x1,
                        y1,
                        x2,
                        y2
                    ),
                    "center": (
                        center_x,
                        center_y
                    ),
                    "confidence": confidence
                }
            )

        return detections

    # ==================================================
    # PERSON TRACKING
    # ==================================================

    def update_person_tracks(self, detections):

        if not detections:

            for person_id in list(
                self.tracks.keys()
            ):

                self.tracks[
                    person_id
                ]["missed"] += 1

            self.remove_old_tracks()

            return

        track_ids = list(
            self.tracks.keys()
        )

        unmatched_tracks = set(
            track_ids
        )

        unmatched_detections = set(
            range(len(detections))
        )

        matches = []

        distances = []

        for person_id in track_ids:

            previous_center = (
                self.tracks[
                    person_id
                ]["center"]
            )

            for (
                detection_index,
                detection
            ) in enumerate(detections):

                current_center = (
                    detection["center"]
                )

                dx = (
                    current_center[0]
                    -
                    previous_center[0]
                )

                dy = (
                    current_center[1]
                    -
                    previous_center[1]
                )

                distance = (
                    dx * dx +
                    dy * dy
                ) ** 0.5

                distances.append(
                    (
                        distance,
                        person_id,
                        detection_index
                    )
                )

        distances.sort(
            key=lambda item: item[0]
        )

        for (
            distance,
            person_id,
            detection_index
        ) in distances:

            if (
                distance
                >
                self.max_tracking_distance
            ):

                continue

            if (
                person_id
                not in unmatched_tracks
            ):

                continue

            if (
                detection_index
                not in unmatched_detections
            ):

                continue

            matches.append(
                (
                    person_id,
                    detection_index
                )
            )

            unmatched_tracks.remove(
                person_id
            )

            unmatched_detections.remove(
                detection_index
            )

        # ==========================================
        # UPDATE MATCHED TRACKS
        # ==========================================

        for (
            person_id,
            detection_index
        ) in matches:

            detection = detections[
                detection_index
            ]

            old_center = self.tracks[
                person_id
            ]["center"]

            new_center = detection[
                "center"
            ]

            self.tracks[
                person_id
            ]["previous_center"] = old_center

            self.tracks[
                person_id
            ]["center"] = new_center

            self.tracks[
                person_id
            ]["box"] = detection["box"]

            self.tracks[
                person_id
            ]["confidence"] = (
                detection["confidence"]
            )

            self.tracks[
                person_id
            ]["missed"] = 0

            trail = self.tracks[
                person_id
            ]["trail"]

            trail.append(
                new_center
            )

            if (
                len(trail)
                >
                self.max_trail_length
            ):

                trail.pop(0)

        # ==========================================
        # UNMATCHED TRACKS
        # ==========================================

        for person_id in unmatched_tracks:

            self.tracks[
                person_id
            ]["missed"] += 1

        # ==========================================
        # NEW TRACKS
        # ==========================================

        for detection_index in (
            unmatched_detections
        ):

            detection = detections[
                detection_index
            ]

            center = detection[
                "center"
            ]

            person_id = (
                self.next_person_id
            )

            self.next_person_id += 1

            self.tracks[
                person_id
            ] = {

                "center": center,

                "previous_center": center,

                "box": detection["box"],

                "confidence": (
                    detection["confidence"]
                ),

                "trail": [
                    center
                ],

                "missed": 0,

                "flow_direction_history": [],

                "flow_direction": "--",

                "flow_magnitude": 0.0,

                "flow_vector": (
                    0.0,
                    0.0
                ),

                "counting_line_side": None,

                "last_crossing_time": 0.0
            }

        self.remove_old_tracks()

    # ==================================================
    # REMOVE OLD TRACKS
    # ==================================================

    def remove_old_tracks(self):

        remove_ids = []

        for (
            person_id,
            track
        ) in self.tracks.items():

            if track["missed"] > 5:

                remove_ids.append(
                    person_id
                )

        for person_id in remove_ids:

            del self.tracks[
                person_id
            ]

    # ==================================================
    # OPTICAL FLOW
    # ==================================================

    def calculate_optical_flow(self, frame):

        start = time.perf_counter()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # ==========================================
        # FIRST FRAME
        # ==========================================

        if self.prev_gray is None:

            self.prev_gray = gray

            self.current_flow = None

            self.flow_magnitude = None

            self.flow_angle = None

            self.direction = "--"

            self.average_motion = 0.0

            self.motion_intensity = "LOW"

            self.flow_consistency = 0.0

            return (
                None,
                0.0,
                "--"
            )

        # ==========================================
        # CALCULATE OPTICAL FLOW
        # ==========================================

        flow = None

        if (
            self.device == "GPU"
            and
            self.cuda_flow_available
            and
            self.flow_gpu is not None
        ):

            try:

                if (
                    self.previous_gpu is None
                    or
                    self.current_gpu is None
                ):

                    self.previous_gpu = (
                        cv2.cuda_GpuMat()
                    )

                    self.current_gpu = (
                        cv2.cuda_GpuMat()
                    )

                self.previous_gpu.upload(
                    self.prev_gray
                )

                self.current_gpu.upload(
                    gray
                )

                flow_gpu = (
                    self.flow_gpu.calc(
                        self.previous_gpu,
                        self.current_gpu,
                        None
                    )
                )

                flow = flow_gpu.download()

            except Exception as e:

                print(
                    "CUDA optical flow failed:",
                    e
                )

                flow = None

        # ==========================================
        # CPU FALLBACK
        # ==========================================

        if flow is None:

            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray,
                gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0
            )

        self.prev_gray = gray

        # ==========================================
        # MAGNITUDE + ANGLE
        # ==========================================

        magnitude, angle = cv2.cartToPolar(
            flow[..., 0],
            flow[..., 1],
            angleInDegrees=True
        )

        self.current_flow = flow

        self.flow_magnitude = magnitude

        self.flow_angle = angle

        # ==========================================
        # AVERAGE MOTION
        # ==========================================

        motion_mask = (
            magnitude
            >
            self.average_motion_threshold
        )

        valid_motion = magnitude[
            motion_mask
        ]

        if len(valid_motion) > 0:

            average_motion = float(
                np.mean(valid_motion)
            )

        else:

            average_motion = 0.0

        self.average_motion = (
            average_motion
        )

        # ==========================================
        # MOTION INTENSITY
        # ==========================================

        motion_intensity = (
            self.calculate_motion_intensity(
                average_motion
            )
        )

        # ==========================================
        # GLOBAL DIRECTION
        # ==========================================

        direction = (
            self.get_dominant_direction(
                magnitude,
                angle
            )
        )

        # IMPORTANT
        self.direction = direction

        # ==========================================
        # FLOW CONSISTENCY
        # ==========================================

        flow_consistency = (
            self.calculate_flow_consistency(
                magnitude,
                angle
            )
        )

        # ==========================================
        # PERFORMANCE
        # ==========================================

        self.last_flow_time = (
            time.perf_counter()
            -
            start
        ) * 1000.0

        self.motion_intensity = (
            motion_intensity
        )

        self.flow_consistency = (
            flow_consistency
        )

        return (
            flow,
            average_motion,
            direction
        )

    # ==================================================
    # MOTION INTENSITY
    # ==================================================

    def calculate_motion_intensity(
        self,
        average_motion
    ):

        if average_motion < 1.5:

            return "LOW"

        elif average_motion < 4.0:

            return "MEDIUM"

        else:

            return "HIGH"

    # ==================================================
    # FLOW CONSISTENCY
    # ==================================================

    def calculate_flow_consistency(
        self,
        magnitude,
        angle
    ):

        mask = (
            magnitude
            >
            self.person_flow_threshold
        )

        if not np.any(mask):

            return 0.0

        valid_magnitude = (
            magnitude[mask]
        )

        valid_angle = (
            angle[mask]
        )

        radians = np.deg2rad(
            valid_angle
        )

        vector_x = np.cos(
            radians
        )

        vector_y = np.sin(
            radians
        )

        mean_x = np.average(
            vector_x,
            weights=valid_magnitude
        )

        mean_y = np.average(
            vector_y,
            weights=valid_magnitude
        )

        mean_length = (
            mean_x * mean_x
            +
            mean_y * mean_y
        ) ** 0.5

        consistency = (
            mean_length * 100.0
        )

        consistency = max(
            0.0,
            min(
                100.0,
                consistency
            )
        )

        return float(
            consistency
        )

    # ==================================================
    # PERSON-REGION OPTICAL FLOW
    # ==================================================

    def calculate_person_flow(
        self,
        track
    ):

        if (
            self.current_flow is None
            or
            self.flow_magnitude is None
            or
            self.flow_angle is None
        ):

            return

        height, width = (
            self.flow_magnitude.shape
        )

        x1, y1, x2, y2 = (
            track["box"]
        )

        x1 = max(
            0,
            min(width - 1, x1)
        )

        x2 = max(
            0,
            min(width, x2)
        )

        y1 = max(
            0,
            min(height - 1, y1)
        )

        y2 = max(
            0,
            min(height, y2)
        )

        if x2 <= x1 or y2 <= y1:

            return

        person_flow = (
            self.current_flow[
                y1:y2,
                x1:x2
            ]
        )

        person_magnitude = (
            self.flow_magnitude[
                y1:y2,
                x1:x2
            ]
        )

        person_angle = (
            self.flow_angle[
                y1:y2,
                x1:x2
            ]
        )

        mask = (
            person_magnitude
            >
            self.person_flow_threshold
        )

        if (
            np.count_nonzero(mask)
            <
            self.min_person_flow_vectors
        ):

            track[
                "flow_magnitude"
            ] = 0.0

            track[
                "flow_vector"
            ] = (
                0.0,
                0.0
            )

            self.update_person_direction(
                track,
                "Stationary"
            )

            return

        valid_magnitude = (
            person_magnitude[mask]
        )

        valid_angle = (
            person_angle[mask]
        )

        average_magnitude = float(
            np.mean(
                valid_magnitude
            )
        )

        radians = np.deg2rad(
            valid_angle
        )

        flow_x = np.cos(
            radians
        )

        flow_y = np.sin(
            radians
        )

        mean_x = np.average(
            flow_x,
            weights=valid_magnitude
        )

        mean_y = np.average(
            flow_y,
            weights=valid_magnitude
        )

        track[
            "flow_magnitude"
        ] = average_magnitude

        track[
            "flow_vector"
        ] = (
            mean_x,
            mean_y
        )

        if (
            average_magnitude
            <
            self.stationary_threshold
        ):

            detected_direction = (
                "Stationary"
            )

        elif (
            abs(mean_x) < 0.15
            and
            abs(mean_y) < 0.15
        ):

            detected_direction = (
                track[
                    "flow_direction"
                ]
            )

        elif abs(mean_x) > abs(mean_y):

            if mean_x > 0:

                detected_direction = "→"

            else:

                detected_direction = "←"

        else:

            if mean_y > 0:

                detected_direction = "↓"

            else:

                detected_direction = "↑"

        self.update_person_direction(
            track,
            detected_direction
        )

    # ==================================================
    # PERSON DIRECTION SMOOTHING
    # ==================================================

    def update_person_direction(
        self,
        track,
        detected_direction
    ):

        history = track[
            "flow_direction_history"
        ]

        history.append(
            detected_direction
        )

        if (
            len(history)
            >
            self.person_direction_history_length
        ):

            history.pop(0)

        counts = {}

        for direction in history:

            counts[direction] = (
                counts.get(
                    direction,
                    0
                )
                +
                1
            )

        if not counts:

            return

        stable_direction = max(
            counts,
            key=counts.get
        )

        stable_count = counts[
            stable_direction
        ]

        if stable_count >= (
            self.person_direction_history_length
            // 2
        ):

            track[
                "flow_direction"
            ] = stable_direction

    # ==================================================
    # UPDATE PERSON FLOW
    # ==================================================

    def update_person_flow(self):

        if self.current_flow is None:

            return

        for (
            person_id,
            track
        ) in self.tracks.items():

            if track["missed"] > 2:

                continue

            self.calculate_person_flow(
                track
            )

    # ==================================================
    # GLOBAL DOMINANT DIRECTION
    # ==================================================

    def get_dominant_direction(
        self,
        magnitude,
        angle
    ):

        mask = (
            magnitude
            >
            self.person_flow_threshold
        )

        if not np.any(mask):

            return self.current_direction

        valid_magnitude = (
            magnitude[mask]
        )

        valid_angles = (
            angle[mask]
        )

        average_magnitude = np.mean(
            valid_magnitude
        )

        if (
            average_magnitude
            <
            self.stationary_threshold
        ):

            detected_direction = (
                "Stationary"
            )

        else:

            radians = np.deg2rad(
                valid_angles
            )

            x = np.cos(
                radians
            )

            y = np.sin(
                radians
            )

            mean_x = np.average(
                x,
                weights=valid_magnitude
            )

            mean_y = np.average(
                y,
                weights=valid_magnitude
            )

            if (
                abs(mean_x) < 0.15
                and
                abs(mean_y) < 0.15
            ):

                detected_direction = (
                    self.current_direction
                )

            elif abs(mean_x) > abs(mean_y):

                if mean_x > 0:

                    detected_direction = "→"

                else:

                    detected_direction = "←"

            else:

                if mean_y > 0:

                    detected_direction = "↓"

                else:

                    detected_direction = "↑"

        self.direction_history.append(
            detected_direction
        )

        if (
            len(
                self.direction_history
            )
            >
            self.direction_history_length
        ):

            self.direction_history.pop(0)

        counts = {}

        for direction in (
            self.direction_history
        ):

            counts[direction] = (
                counts.get(
                    direction,
                    0
                )
                +
                1
            )

        if counts:

            stable_direction = max(
                counts,
                key=counts.get
            )

            stable_count = counts[
                stable_direction
            ]

            if stable_count >= (
                self.direction_history_length
                // 2
            ):

                self.current_direction = (
                    stable_direction
                )

        return self.current_direction

    # ==================================================
    # DENSITY
    # ==================================================

    def calculate_density(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        image_area = (
            width * height
        )

        if image_area <= 0:

            return 0.0

        # Normalize to 100,000 pixels.
        #
        # This is image-space density.
        # It is NOT people/m² because the
        # camera has not been physically calibrated.

        density = (
            self.people_count
            /
            image_area
        ) * 100000.0

        return float(
            density
        )

    # ==================================================
    # OCCUPANCY
    # ==================================================

    def calculate_occupancy(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        image_area = (
            width * height
        )

        if image_area <= 0:

            return 0.0

        total_person_area = 0

        for (
            person_id,
            track
        ) in self.tracks.items():

            if track["missed"] > 2:

                continue

            x1, y1, x2, y2 = (
                track["box"]
            )

            x1 = max(
                0,
                min(width, x1)
            )

            x2 = max(
                0,
                min(width, x2)
            )

            y1 = max(
                0,
                min(height, y1)
            )

            y2 = max(
                0,
                min(height, y2)
            )

            box_width = max(
                0,
                x2 - x1
            )

            box_height = max(
                0,
                y2 - y1
            )

            total_person_area += (
                box_width *
                box_height
            )

        occupancy = (
            total_person_area
            /
            image_area
        ) * 100.0

        # Prevent overlapping bounding boxes from
        # producing an impossible percentage.

        occupancy = min(
            100.0,
            occupancy
        )

        return float(
            occupancy
        )

    # ==================================================
    # FLOW CROSSING DETECTION
    # ==================================================

    def calculate_people_flow(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        if height <= 0:

            return 0.0

        counting_line_y = int(
            height *
            self.counting_line_ratio
        )

        current_time = time.perf_counter()

        new_crossings = 0

        for (
            person_id,
            track
        ) in self.tracks.items():

            if track["missed"] > 0:

                continue

            previous_center = (
                track["previous_center"]
            )

            current_center = (
                track["center"]
            )

            previous_y = (
                previous_center[1]
            )

            current_y = (
                current_center[1]
            )

            previous_side = (
                previous_y
                <
                counting_line_y
            )

            current_side = (
                current_y
                <
                counting_line_y
            )

            if (
                previous_side
                !=
                current_side
            ):

                last_crossing = (
                    track[
                        "last_crossing_time"
                    ]
                )

                # Prevent the same tracked person
                # from generating repeated counts
                # while moving around the line.

                if (
                    current_time
                    -
                    last_crossing
                    >
                    1.0
                ):

                    new_crossings += 1

                    track[
                        "last_crossing_time"
                    ] = current_time

        if new_crossings > 0:

            self.flow_crossing_history.append(
                (
                    current_time,
                    new_crossings
                )
            )

        cutoff_time = (
            current_time
            -
            self.flow_time_window
        )

        self.flow_crossing_history = [
            item
            for item in
            self.flow_crossing_history
            if item[0] >= cutoff_time
        ]

        total_crossings = sum(
            count
            for (
                timestamp,
                count
            ) in self.flow_crossing_history
        )

        if self.flow_time_window > 0:

            flow_rate = (
                total_crossings
                /
                self.flow_time_window
            )

        else:

            flow_rate = 0.0

        self.flow_crossings = (
            total_crossings
        )

        return float(
            flow_rate
        )

    # ==================================================
    # ZONE-BASED CONGESTION
    # ==================================================

    def calculate_congestion(
        self,
        frame
    ):

        height, width = frame.shape[:2]

        if (
            height <= 0
            or
            width <= 0
        ):

            return "LOW"

        rows = self.zone_rows
        cols = self.zone_cols

        zone_people = np.zeros(
            (rows, cols),
            dtype=np.int32
        )

        zone_person_area = np.zeros(
            (rows, cols),
            dtype=np.float64
        )

        # ==========================================
        # Assign people to zones
        # ==========================================

        for (
            person_id,
            track
        ) in self.tracks.items():

            if track["missed"] > 2:

                continue

            center_x, center_y = (
                track["center"]
            )

            col = int(
                center_x
                /
                width
                *
                cols
            )

            row = int(
                center_y
                /
                height
                *
                rows
            )

            col = max(
                0,
                min(cols - 1, col)
            )

            row = max(
                0,
                min(rows - 1, row)
            )

            zone_people[
                row,
                col
            ] += 1

            x1, y1, x2, y2 = (
                track["box"]
            )

            x1 = max(
                0,
                min(width, x1)
            )

            x2 = max(
                0,
                min(width, x2)
            )

            y1 = max(
                0,
                min(height, y1)
            )

            y2 = max(
                0,
                min(height, y2)
            )

            box_area = (
                max(0, x2 - x1)
                *
                max(0, y2 - y1)
            )

            zone_person_area[
                row,
                col
            ] += box_area

        # ==========================================
        # Calculate zone congestion
        # ==========================================

        congested_zones = 0
        people_in_congested_zones = 0
        for row in range(rows):

            for col in range(cols):

                people = int(
                    zone_people[
                        row,
                        col
                    ]
                )

                zone_area = (
                    (
                        width
                        /
                        cols
                    )
                    *
                    (
                        height
                        /
                        rows
                    )
                )

                if zone_area <= 0:

                    continue

                occupancy = (
                    zone_person_area[
                        row,
                        col
                    ]
                    /
                    zone_area
                ) * 100.0

                # ----------------------------------
                # Optical-flow motion in this zone
                # ----------------------------------

                if (
                    self.flow_magnitude
                    is not None
                ):

                    y1 = int(
                        row
                        *
                        height
                        /
                        rows
                    )

                    y2 = int(
                        (row + 1)
                        *
                        height
                        /
                        rows
                    )

                    x1 = int(
                        col
                        *
                        width
                        /
                        cols
                    )

                    x2 = int(
                        (col + 1)
                        *
                        width
                        /
                        cols
                    )

                    zone_flow = (
                        self.flow_magnitude[
                            y1:y2,
                            x1:x2
                        ]
                    )

                    motion_mask = (
                        zone_flow
                        >
                        self.person_flow_threshold
                    )

                    if np.any(
                        motion_mask
                    ):

                        motion_values = (
                            zone_flow[
                                motion_mask
                            ]
                        )

                        zone_motion = float(
                            np.mean(
                                motion_values
                            )
                        )

                    else:

                        zone_motion = 0.0

                else:

                    zone_motion = 0.0

                # ==================================
                # CONGESTION CONDITION
                # ==================================
                #
                # A zone is considered congested
                # when:
                #
                # 1. At least a few people are present
                # 2. Occupancy is relatively high
                # 3. Optical-flow motion is low
                #
                # This keeps optical flow central to
                # congestion analysis.

                if (
                    people
                    >=
                    self.zone_density_threshold
                    and
                    occupancy
                    >=
                    self.zone_occupancy_threshold
                    and
                    zone_motion
                    <
                    self.zone_motion_threshold
                ):

                    congested_zones += 1

                    people_in_congested_zones += people

        self.congested_zones = (
            congested_zones
        )
        # ==========================================
        # CONGESTION SHARE
        # ==========================================

        if self.total_zones > 0:

            self.congestion_share = (
            congested_zones
            /
            self.total_zones
        ) * 100.0

        else:

            self.congestion_share = 0.0

        # ==========================================
        # PEOPLE IN CONGESTED ZONES
        # ==========================================

        total_active_people = 0

        for track in self.tracks.values():

            if track["missed"] <= 2:

                total_active_people += 1

        if total_active_people > 0:

            self.people_congestion_share = (
                people_in_congested_zones
                /
                total_active_people
            ) * 100.0

        else:

            self.people_congestion_share = 0.0

        self.people_in_congested_zones = (
            people_in_congested_zones
        )
        # ==========================================
        # GLOBAL CONGESTION LEVEL
        # ==========================================

        congestion_ratio = (
            congested_zones
            /
            self.total_zones
        )

        if congestion_ratio >= 0.34:

            congestion = "HIGH"

        elif congestion_ratio >= 0.12:

            congestion = "MEDIUM"

        else:

            congestion = "LOW"

        return congestion

    # ==================================================
    # DRAW GLOBAL OPTICAL FLOW
    # ==================================================

    def draw_optical_flow(
        self,
        frame,
        flow,
        step=48
    ):

        if flow is None:

            return frame

        output = frame.copy()

        height, width = (
            flow.shape[:2]
        )

        for y in range(
            step // 2,
            height,
            step
        ):

            for x in range(
                step // 2,
                width,
                step
            ):

                fx, fy = flow[y, x]

                magnitude = (
                    fx * fx +
                    fy * fy
                ) ** 0.5

                if (
                    magnitude
                    <
                    self.person_flow_threshold
                ):

                    continue

                end_x = int(
                    x + fx * 3
                )

                end_y = int(
                    y + fy * 3
                )

                end_x = max(
                    0,
                    min(
                        width - 1,
                        end_x
                    )
                )

                end_y = max(
                    0,
                    min(
                        height - 1,
                        end_y
                    )
                )

                cv2.arrowedLine(
                    output,
                    (x, y),
                    (end_x, end_y),
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                    tipLength=0.3
                )

        return output

    # ==================================================
    # DRAW COUNTING LINE
    # ==================================================

    def draw_counting_line(
        self,
        frame
    ):

        output = frame.copy()

        height, width = (
            output.shape[:2]
        )

        line_y = int(
            height *
            self.counting_line_ratio
        )

        cv2.line(
            output,
            (0, line_y),
            (width, line_y),
            (255, 255, 0),
            1,
            cv2.LINE_AA
        )

        return output

    # ==================================================
    # DRAW PEOPLE
    # ==================================================

    def draw_people(
        self,
        frame
    ):

        output = frame.copy()

        for (
            person_id,
            track
        ) in self.tracks.items():

            if track["missed"] > 2:

                continue

            x1, y1, x2, y2 = (
                track["box"]
            )

            confidence = (
                track["confidence"]
            )

            center = (
                track["center"]
            )

            # ==========================================
            # PERSON BOX
            # ==========================================

            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ==========================================
            # PERSON ID
            # ==========================================

            label = (
                f"P{person_id} "
                f"{confidence:.2f}"
            )

            cv2.putText(
                output,
                label,
                (
                    x1,
                    max(
                        y1 - 8,
                        20
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            # ==========================================
            # TRACK TRAIL
            # ==========================================

            trail = track[
                "trail"
            ]

            if len(trail) >= 2:

                for i in range(
                    1,
                    len(trail)
                ):

                    cv2.line(
                        output,
                        trail[i - 1],
                        trail[i],
                        (255, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

            # ==========================================
            # PERSON OPTICAL FLOW ARROW
            # ==========================================

            mean_x, mean_y = (
                track[
                    "flow_vector"
                ]
            )

            flow_magnitude = (
                track[
                    "flow_magnitude"
                ]
            )

            if (
                flow_magnitude
                >=
                self.person_flow_threshold
            ):

                vector_length = (
                    mean_x * mean_x
                    +
                    mean_y * mean_y
                ) ** 0.5

                if vector_length > 0.01:

                    arrow_length = min(
                        max(
                            flow_magnitude * 12,
                            25
                        ),
                        80
                    )

                    end_x = int(
                        center[0]
                        +
                        (
                            mean_x
                            /
                            vector_length
                        )
                        *
                        arrow_length
                    )

                    end_y = int(
                        center[1]
                        +
                        (
                            mean_y
                            /
                            vector_length
                        )
                        *
                        arrow_length
                    )

                    height, width = (
                        output.shape[:2]
                    )

                    end_x = max(
                        0,
                        min(
                            width - 1,
                            end_x
                        )
                    )

                    end_y = max(
                        0,
                        min(
                            height - 1,
                            end_y
                        )
                    )

                    cv2.arrowedLine(
                        output,
                        center,
                        (
                            end_x,
                            end_y
                        ),
                        (0, 165, 255),
                        3,
                        cv2.LINE_AA,
                        tipLength=0.3
                    )

            # ==========================================
            # CENTER POINT
            # ==========================================

            cv2.circle(
                output,
                center,
                3,
                (0, 165, 255),
                -1
            )

        return output

    # ==================================================
    # MAIN PROCESS
    # ==================================================

    def process(self, frame):

        self.frame_counter += 1

        # ==========================================
        # PERSON DETECTION
        # ==========================================

        if (
            self.frame_counter == 1
            or
            self.frame_counter
            %
            self.detection_interval
            == 0
        ):

            self.last_detections = (
                self.detect_people(
                    frame
                )
            )

            self.update_person_tracks(
                self.last_detections
            )

        # ==========================================
        # OPTICAL FLOW
        # ==========================================

        (
            flow,
            average_motion,
            direction
        ) = self.calculate_optical_flow(
            frame
        )

        # ==========================================
        # PERSON-SPECIFIC OPTICAL FLOW
        # ==========================================

        self.update_person_flow()

        # ==========================================
        # PEOPLE COUNT
        # ==========================================

        self.people_count = sum(
            1
            for track in self.tracks.values()
            if track["missed"] <= 2
        )

        # ==========================================
        # DENSITY
        # ==========================================

        self.density = (
            self.calculate_density(
                frame
            )
        )

        # ==========================================
        # OCCUPANCY
        # ==========================================

        self.occupancy = (
            self.calculate_occupancy(
                frame
            )
        )

        # ==========================================
        # PEOPLE FLOW
        # ==========================================

        self.people_flow = (
            self.calculate_people_flow(
                frame
            )
        )

        # ==========================================
        # CONGESTION
        # ==========================================

        self.congestion = (
            self.calculate_congestion(
                frame
            )
        )

        # ==========================================
        # DRAW OUTPUT
        # ==========================================

        output_frame = frame.copy()

        # Global optical-flow arrows
        output_frame = (
            self.draw_optical_flow(
                output_frame,
                flow
            )
        )

        # People boxes, IDs, trails and
        # person-specific optical-flow arrows
        output_frame = (
            self.draw_people(
                output_frame
            )
        )

        # Counting line
        output_frame = (
            self.draw_counting_line(
                output_frame
            )
        )

        # ==========================================
        # RETURN RESULTS
        # ==========================================

        return {

            "frame":
                output_frame,

            "people_count":
                self.people_count,

            "density":
                self.density,

            "direction":
                self.direction,

            "average_motion":
                self.average_motion,

            "motion_intensity":
                self.motion_intensity,

            "flow_consistency":
                self.flow_consistency,

            "occupancy":
                self.occupancy,

            "flow":
                self.people_flow,

            "congestion":
                self.congestion,

            "congested_zones":
                self.congested_zones,

            "congestion_share":
                self.congestion_share,

            "people_in_congested_zones":
                self.people_in_congested_zones,

            "people_congestion_share":
                self.people_congestion_share,

            "yolo_time":
                self.last_yolo_time,

            "flow_time":
                self.last_flow_time
        }

    # ==================================================
    # RESET
    # ==================================================

    def reset(self):

        self.prev_gray = None

        self.previous_gpu = None

        self.current_gpu = None

        self.current_flow = None

        self.flow_magnitude = None

        self.flow_angle = None

        self.frame_counter = 0

        self.last_detections = []

        self.next_person_id = 1

        self.tracks = OrderedDict()

        self.people_count = 0

        self.density = 0.0

        self.direction = "--"

        self.average_motion = 0.0

        self.motion_intensity = "--"

        self.flow_consistency = 0.0

        self.occupancy = 0.0

        self.people_flow = 0.0

        self.flow_crossings = 0

        self.flow_crossing_history = []

        self.congestion = "LOW"

        self.congested_zones = 0

        self.congestion_share = 0.0

        self.people_in_congested_zones = 0

        self.people_congestion_share = 0.0

        self.direction_history = []

        self.current_direction = "--"

        self.last_yolo_time = 0.0

        self.last_flow_time = 0.0

