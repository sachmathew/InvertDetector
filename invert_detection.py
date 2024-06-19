from collections import deque
import cv2
import numpy as np
from tqdm import tqdm
import math


class InvertDetector:
    def __init__(self, video: str, outpath: str = '', fgbg: str = 'CNT', args: list[int, int] = [8, 480]):
        self.OUTPATH = outpath
        self.MID_BRIGHTNESS = 128.0
        self.BASE_STDDV = 42.5
        self.SECONDS_WINDOW = 0.21
        self.FGBG_TYPE = fgbg
        self.FGBG_ARGS = args

        self._load_raw_video(video)

    def set_params(self, b, o):
        self.light_blur = b
        self.open_threshold = o

    def _load_raw_video(self, video: str):
        self.VIDEO = video
        self.CAP = cv2.VideoCapture(self.VIDEO)
        self.FPS = self.CAP.get(cv2.CAP_PROP_FPS)
        self.WINDOW_LENGTH = int(self.FPS*self.SECONDS_WINDOW)
        self.TOTAL = int(self.CAP.get(cv2.CAP_PROP_FRAME_COUNT))
        self.WIDTH = int(self.CAP.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.HEIGHT = int(self.CAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
        min_side = min(self.WIDTH, self.HEIGHT)
        self.light_blur = round(5.0*972/min_side)
        self.open_threshold = round(5.0*972/min_side)
        # tot_sec = self.TOTAL/self.FPS
        # h = int(tot_sec/3600)
        # m = int((tot_sec % 3600)/60)
        # s = int(tot_sec%60)
        # print(f'loaded {self.VIDEO}: FPS={self.FPS}, WINDOW_FRAMES={self.WINDOW_LENGTH}, LENGTH={h}:{m:02d}:{s:02d}, (w, h)=({self.WIDTH}, {self.HEIGHT})')
        
    # def process_full(self):
    #     self.LIGHT_RESULT
    #     self.MASK_NORM_RESULT = cv2.VideoWriter(self.OUTPATH+'mask_norm.mp4',
    #                                             cv2.VideoWriter_fourcc(*'mp4v'),
    #                                             self.FPS, (self.WIDTH, self.HEIGHT))
    #     self.MASK_RAW_RESULT = cv2.VideoWriter(self.OUTPATH+'mask_raw.mp4',
    #                                            cv2.VideoWriter_fourcc(*'mp4v'),
    #                                            self.FPS, (self.WIDTH, self.HEIGHT))
    #     for i, _raw, _norm, _mask in tqdm(self.process(), total=self.TOTAL):
    #         self._norm_circled = cv2.cvtColor(_norm, cv2.COLOR_GRAY2BGR)
    #         self._raw_circled = _raw
    #         self._mask_contours, _ = cv2.findContours(_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #         cv2.drawContours(self._norm_circled, self._mask_contours, -1, (0, 0, 255), -1)
    #         cv2.drawContours(self._raw_circled, self._mask_contours, -1, (0, 0, 255), -1)
    #         self.MASK_NORM_RESULT.write(self._norm_circled)
    #         self.MASK_RAW_RESULT.write(self._raw_circled)
    #     clear(self._norm_circled, self._raw_circled, self._mask_contours)
    #     self.MASK_NORM_RESULT.release()
    #     self.MASK_RAW_RESULT.release()
    #     with open(self.OUTPATH+'data.csv', 'w', encoding='utf-8-sig') as f:
    #         pd.DataFrame(self.data).to_csv(f)
    #     print('done saving from InvertDetector')

    def process(self):
        self._frame_counter = 0
        self._agg = None
        self._window = deque([])
        self._fgbg = gen_fgbg(self.FGBG_TYPE, self.FGBG_ARGS)
        self.data = {}
        while self._frame_counter < self.TOTAL:
            self._ret, self._raw = self.CAP.read()
            if not self._ret:
                print('break')
                break
            # self._grey = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
            self._norm = self._normalize_pass(self._raw)
            self._mask = self._mask_pass(self._norm)
            self._csv_pass(self._frame_counter, self._mask)
            yield self._frame_counter, self._raw, self._norm, self._mask
            self._frame_counter += 1
        self.CAP.release()
        clear(self._ret, self._agg, self._window, self._fgbg,
              self._raw, self._norm, self._mask,  # self._grey
              self._to_norm, self._light, self._flat, self._mean, self._stdv,
              self._to_mask, self._differenced,
              self._contours)

    def process_full(self):
        self.LIGHT_RESULT = cv2.VideoWriter(self.OUTPATH+'light.mp4',
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            self.FPS, (self.WIDTH, self.HEIGHT))
        self.NORM_RESULT = cv2.VideoWriter(self.OUTPATH+'norm.mp4',
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           self.FPS, (self.WIDTH, self.HEIGHT))
        self.DIFF_RESULT = cv2.VideoWriter(self.OUTPATH+'diff.mp4',
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           self.FPS, (self.WIDTH, self.HEIGHT))
        self.MASK_RESULT = cv2.VideoWriter(self.OUTPATH+'mask.mp4',
                                           cv2.VideoWriter_fourcc(*'mp4v'),
                                           self.FPS, (self.WIDTH, self.HEIGHT))
        self._frame_counter = 0
        self._agg = None
        self._window = deque([])
        self._fgbg = gen_fgbg(self.FGBG_TYPE, self.FGBG_ARGS)
        self.data = {}
        for self._frame_counter in tqdm(range(self.TOTAL)):
            self._ret, self._raw = self.CAP.read()
            if not self._ret:
                print('break')
                break
            # self._grey = cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY)
            self._lighting, self._norm = self._normalize_pass(self._raw, True)
            self._diffed, self._mask = self._mask_pass(self._norm, True)
            self._csv_pass(self._frame_counter, self._mask)
            self.LIGHT_RESULT.write(self._lighting)
            self.NORM_RESULT.write(cv2.cvtColor(self._norm, cv2.COLOR_GRAY2BGR))
            self.DIFF_RESULT.write(cv2.cvtColor(self._diffed, cv2.COLOR_GRAY2BGR))
            self.MASK_RESULT.write(cv2.cvtColor(self._mask, cv2.COLOR_GRAY2BGR))
        self.CAP.release()
        self.LIGHT_RESULT.release()
        self.NORM_RESULT.release()
        self.DIFF_RESULT.release()
        self.MASK_RESULT.release()
        clear(self._ret, self._agg, self._window, self._fgbg,
              self._raw, self._lighting, self._norm, self._diffed, self._mask,  # self._grey
              self._to_norm, self._light, self._flat, self._mean, self._stdv,
              self._to_mask, self._differenced,
              self._contours)

    def _normalize_pass(self, to_norm, full=False):
        self._to_norm = to_norm.astype(np.float32)
        self._window.append(self._to_norm)
        if isinstance(self._agg, np.ndarray):
            self._agg += self._to_norm
        else:
            self._agg = self._to_norm.copy()
        if self._frame_counter > self.WINDOW_LENGTH:
            self._agg -= self._window.popleft()

        self._light = cv2.blur(self._agg/len(self._window), (self.light_blur, self.light_blur))
        self._flat = cv2.cvtColor(self._to_norm - self._light, cv2.COLOR_BGR2GRAY)
        self._mean, self._stdv = cv2.meanStdDev(cv2.blur(self._flat, (self.light_blur, self.light_blur)))
        if full:
            return self._light.astype(np.uint8), np.clip((self._flat-self._mean[0][0])*(self.BASE_STDDV/self._stdv[0][0])+self.MID_BRIGHTNESS, 0, 255).astype(np.uint8)
        return np.clip((self._flat-self._mean[0][0])*(self.BASE_STDDV/self._stdv[0][0])+self.MID_BRIGHTNESS,
                       0, 255).astype(np.uint8)

    def _mask_pass(self, to_mask, full=False):
        self._to_mask = cv2.blur(to_mask, (self.light_blur, self.light_blur))
        self._differenced = self._fgbg.apply(self._to_mask)
        if full:
            return self._differenced, cv2.morphologyEx(self._differenced, cv2.MORPH_OPEN, np.ones((self.open_threshold, self.open_threshold), np.uint8))
        return cv2.morphologyEx(self._differenced, cv2.MORPH_OPEN,
                                np.ones((self.open_threshold, self.open_threshold), np.uint8))

    def _csv_pass(self, frame_num: int, mask):
        self._contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print('real', self._contours)
        # self._centers = get_centers(self._contours)
        # for c in self._centers:      
        self.data[frame_num] = self._contours

    #def segment(self, v, outpath, mask=None, k=5, threshold=205, model=None):
    #    cap = cv2.VideoCapture(v)
    #    result = cv2.VideoWriter(outpath, cv2.VideoWriter_fourcc(*'mp4v'),
    #                             cap.get(cv2.CAP_PROP_FPS), 
    #                             (int(cap.get(3)), int(cap.get(4))))
    #    print("getting fg")
    #    if model is None:
    #        model = torch.load('unet.pth').to('cpu')
    #    model.eval()
    #    with torch.no_grad():
    #        for i in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
    #            ret, img = cap.read()
    #            if not ret:
    #                break
    #            img = np.transpose(img.astype("float32") / 255.0, (2, 0, 1))
    #            img = np.expand_dims(img, 0)
    #            img = torch.from_numpy(img).to('cpu')
    #            predMask = torch.sigmoid(model(img).squeeze()).cpu().numpy()
    #            min = predMask.min()
    #            predMask = (255*(predMask-min)/(predMask.max()-min)).astype(np.uint8)
    #            predMask = (predMask >= threshold) * predMask
    #            predMask = cv2.morphologyEx(predMask, cv2.MORPH_OPEN, 
    #                                        np.ones((k, k), np.uint8))
    #            result.write(cv2.cvtColor(predMask, cv2.COLOR_GRAY2BGR))
    #        cap.release()
    #        result.release() 


def rois_to_snippets(rois: list[tuple[tuple[int, int], int]], contours_in_video) -> dict[tuple[tuple[int, int], int], tuple[int, int]]:
    return get_collision_snippets_by_roi(get_collision_frames_by_roi(rois, contours_in_video))


def roi_to_snippets(roi: list[tuple[tuple[int, int], int]], contours_in_video) -> list[tuple[int, int]]:
    return get_collision_snippets(get_collision_frames(roi, contours_in_video))


def get_collision_snippets_by_roi(collision_frames_by_roi: dict[tuple[tuple[int, int], int], list[int]], on_thresh: int = 3, off_thresh: int = 3) -> dict[tuple[tuple[int, int], int], tuple[int, int]]:
    return {list_to_ROI(roi): get_collision_snippets(collision_frames, on_thresh, off_thresh)
            for roi, collision_frames in collision_frames_by_roi.items()}


def get_collision_snippets(collision_frames: list[int], on_thresh: int = 3, off_thresh: int = 3) -> list[tuple[int, int]]:
    snippets = []
    start_frame = -off_thresh-1
    last_frame = -off_thresh-1
    for frame in collision_frames:
        if frame-last_frame > off_thresh:
            if last_frame-start_frame >= on_thresh:
                snippets += [(start_frame, last_frame)]
            start_frame = frame 
        last_frame = frame
    return snippets


def get_collision_frames_by_roi(rois: list[tuple[tuple[int, int], int]], contours_in_video) -> dict[tuple[tuple[int, int], int], list[int]]:
    return {list_to_ROI(roi): get_collision_frames(roi, contours_in_video) for roi in rois}


def list_to_ROI(l):
    return ((l[0][0], l[0][1]), l[1])


def get_collision_frames(roi: tuple[tuple[int, int], int], contours_in_video) -> list[int]:
    return [frame for frame, contours in contours_in_video.items() if _contours_intersect(roi, contours)]


def _contours_intersect(roi: tuple[tuple[int, int], int], contours) -> bool:
    for contour in contours:
        if _contour_intersect(roi, contour):
            return True
    return False


def _contour_intersect(roi: tuple[tuple[int, int], int], contour) -> bool:
    for i in range(len(contour)-1):
        if _check_line_in_roi(contour[i][0], contour[i+1][0], roi):
            return True
    return False


def _check_line_in_roi(A, B, roi: tuple[tuple[int, int], int]) -> bool:
    return _check_point_in_roi(A, roi) or _check_point_in_roi(B, roi) or _check_point_in_roi(((A[0]+B[0])/2, (A[1]+B[1])/2), roi)


def _check_point_in_roi(point, roi: tuple[tuple[int, int], int]) -> bool:
    return (point[0]-roi[0][0])**2 + (point[1]-roi[0][1])**2 <= roi[1]


def closest_roi_point_is_within(point, rois):
    o = -1
    d = 100000
    for i in range(len(rois)):
        if _check_point_in_roi(point, rois[i]):
            temp = math.sqrt((point[0]-rois[i][0][0])**2 + (point[1]-rois[i][0][1])**2)
            if temp < d:
                d = temp
                o = i
    return o


def gen_fgbg(alg, args=None):
    if not args:
        if alg == 'KNN':
            return cv2.createBackgroundSubtractorKNN()
        if alg == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2()
        if alg == 'MOG':
            return cv2.bgsegm.createBackgroundSubtractorMOG()
        if alg == 'GMG':
            return cv2.bgsegm.createBackgroundSubtractorGMG()
        if alg == 'CNT':
            return cv2.bgsegm.createBackgroundSubtractorCNT()
        if alg == 'GSOC':
            return cv2.bgsegm.createBackgroundSubtractorGSOC()
        if alg == 'LSBP':
            return cv2.bgsegm.createBackgroundSubtractorLSBP()
    else:
        if alg == 'KNN':
            return cv2.createBackgroundSubtractorKNN(args[0], args[1], False)
        if alg == 'MOG2':
            return cv2.createBackgroundSubtractorMOG2(args[0], args[1], False)
        if alg == 'MOG':
            return cv2.bgsegm.createBackgroundSubtractorMOG(args[0], args[1])
        if alg == 'GMG':
            return cv2.bgsegm.createBackgroundSubtractorGMG(args[0], args[1])
        if alg == 'CNT':
            return cv2.bgsegm.createBackgroundSubtractorCNT(args[0], True,
                                                            args[1], True)
        if alg == 'GSOC':
            return cv2.bgsegm.createBackgroundSubtractorGSOC()
        if alg == 'LSBP':
            return cv2.bgsegm.createBackgroundSubtractorLSBP()


def clear(*argv):
    for arg in argv:
        del arg


def get_centers(contours):
    centers = [None]*len(contours)
    for i, c in enumerate(contours):
        M = cv2.moments(c)
        centers[i] = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    return centers


# if __name__ == "__main__":
#     vid = 'raw.mp4'
#     detect = InvertDetector(vid)
#     detect.process_full()
#     print('end main')
