import numpy as np
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk

from puzzle_segmentation import (image_resize, morphology_postprocess, find_polygons,
                                 draw_transarent_polygons, remove_frame_edge_puzzles,
                                 find_marginal_puzzles, distance)


def main():
    tile = cv2.imread("data/tile.jpg")
    cap = cv2.VideoCapture("data/IMG_2855.mp4")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("output/result.mp4", fourcc, 20.0, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = image_resize(img, height=512)
        h, w = img.shape[:2]

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        entropy_img = entropy(gray, disk(3))
        scaled_entropy = entropy_img / entropy_img.max()

        thresh = scaled_entropy > 0.72

        size_ratio = 1.5
        min_size = img.size / (size_ratio * 100) ** 2
        mask = morphology_postprocess(thresh, min_size)

        polygons = find_polygons(mask, n_poly=-1, min_area=500)
        inner_polygons = remove_frame_edge_puzzles(polygons, w, h)

        overlayed_img, _ = draw_transarent_polygons(img, mask, inner_polygons)

        marginal_puzzles, corner_puzzles = find_marginal_puzzles(inner_polygons)
        for shape in marginal_puzzles:
            for v1, v2 in zip(shape, np.roll(shape, 1, axis=0)):
                if distance(v1[0], v2[0]) > 50:
                    cv2.line(overlayed_img, v1[0], v2[0], (255, 0, 0), 2)

        # matches_mask = match_puzzle(tile, img)

        result = image_resize(overlayed_img, width=frame.shape[1], height=frame.shape[0])

        out.write(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        cv2.imshow('frame', cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
