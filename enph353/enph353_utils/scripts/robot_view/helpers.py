import cv2
import numpy as np


class HotPixelAnalyzer:
    def init(self, threshold):
        self.threshold = threshold

    def analyze_image(self, frame_bgr):
        # Convert to RGB
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # --- Compute CMYK magenta channel ---
        rgb_norm = rgb.astype(np.float32) / 255.0
        R, G, B = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]

        C = 1 - R
        M = 1 - G
        Y = 1 - B
        K = np.minimum(np.minimum(C, M), Y)

        # CMYK magenta (M') = M - K
        magenta_intensity = (M - K) * 255
        magenta_intensity = magenta_intensity.astype(np.uint8)

        # Create hot mask
        hot_mask = magenta_intensity > self.threshold
        num_hot = np.count_nonzero(hot_mask)

        percentage_hot_pixels = (num_hot / magenta_intensity.size) * 100
        return percentage_hot_pixels


class CluePublisher:
    def __init__(self, threshold=4):
        self.threshold = threshold
        self.guessCategories = [
            "BANDIT",
            "WEAPON",
            "MOTIVE",
            "PLACE",
            "TIME",
            "CRIME",
            "VICTIM",
            "SIZE",
        ]
        self.map = {
            "BANDIT": {},
            "WEAPON": {},
            "MOTIVE": {},
            "PLACE": {},
            "TIME": {},
            "CRIME": {},
            "VICTIM": {},
            "SIZE": {},
        }

    def addReading(self, clue, reading, cumulativeConfidence):
        if clue not in self.map:
            return
        if not reading in self.map[clue]:
            self.map[clue][reading] = [1, cumulativeConfidence]
            return
        else:
            max_confidence = max(cumulativeConfidence, self.map[clue][reading][1])
            self.map[clue][reading][0] += 1
            self.map[clue][reading][1] = max_confidence

        if self.map[clue][reading][0] >= self.threshold:
            self.publishClue(clue)
            return

    def publishClue(self, clue):
        a = self.guessCategories.pop()
        while True:
            if not self.map[a]:
                self.publish(a, "MR BEAAAAAST")
            else:
                max_reading = max(
                    self.map[a], key=lambda k: (self.map[a][k][0], self.map[a][k][1])
                )
                self.publish(a, max_reading)

            self.map.pop(a)
            if a == clue:
                break
            a = self.guessCategories.pop()

    def publish(self, clue, reading):
        # placeholder for ros
        print("GUESSING A CLUE: ", clue, reading)


class ClueTracker:
    def __init__(self, threshold=4):
        self.threshold = threshold
        self.guessCategories = [
            "BANDIT",
            "WEAPON",
            "MOTIVE",
            "PLACE",
            "TIME",
            "CRIME",
            "VICTIM",
            "SIZE",
        ]
        self.map = {
            "BANDIT": {},
            "WEAPON": {},
            "MOTIVE": {},
            "PLACE": {},
            "TIME": {},
            "CRIME": {},
            "VICTIM": {},
            "SIZE": {},
        }

        self.first_clues = {}

    def addReading(self, clue, reading, cumulativeConfidence):
        if clue not in self.map:
            return
        if not reading in self.map[clue]:
            self.map[clue][reading] = 1
            return
        else:
            # max_confidence = max(cumulativeConfidence, self.map[clue][reading][1])
            self.map[clue][reading] += 1
            # self.map[clue][reading][1] = max_confidence

        if (
            self.map[clue][reading] >= self.threshold
            and clue not in self.first_clues.keys()
        ):
            self.first_clues[clue] = reading
            print("FIRST CLUE FOUND: TYPE: ", clue, ", CLUE: ", reading)
            # self.publishClue(clue)
            return

    def get_first_clues(self):
        return self.first_clues.copy()

    def get_found_clue_count(self):
        return len(self.first_clues)

    def get_frequent_clues(self):
        print("CURRENT MAP: ", self.map)

        most_frequent = {
            clue_type: max(clue_histogram.items(), key=lambda kv: kv[1])[0]
            for clue_type, clue_histogram in self.map.items()
            if clue_histogram
        }

        return most_frequent

    # def publishClue(self, clue):
    #     a = self.guessCategories.pop()
    #     while True:
    #         if not self.map[a]:
    #             self.publish(a, "MR BEAAAAAST")
    #         else:
    #             max_reading = max(
    #                 self.map[a], key=lambda k: (self.map[a][k][0], self.map[a][k][1])
    #             )
    #             self.publish(a, max_reading)

    #         self.map.pop(a)
    #         if a == clue:
    #             break
    #         a = self.guessCategories.pop()

    # def publish(self, clue, reading):
    #     # placeholder for ros
    #     print("GUESSING A CLUE: ", clue, reading)
