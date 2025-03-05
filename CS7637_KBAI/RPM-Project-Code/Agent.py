from PIL import Image, ImageFilter, ImageChops
import numpy as np
import cv2
import time
# import matplotlib as plt

class Agent:
    def __init__(self):
        # Initialize an empty list to store solution times for each problem
        self.imgA = None
        self.imgB = None
        self.imgC = None
        self.imgD = None
        self.imgE = None
        self.imgF = None
        self.imgG = None
        self.imgH = None
        self.solution_times = []
        self.dpr_weight = 1
        self.ipr_weight = 1
        self.incorrect_answers = {
            "Basic Problem B-04": 7,
            "Basic Problem B-05": 8,
            "Basic Problem B-06": 1,
            "Basic Problem B-09": 1,
            "Basic Problem B-10": 7,
            "Basic Problem B-11": 4,
            "Basic Problem C-02": 10,
            "Basic Problem C-03": 10,
            "Basic Problem C-07": 7,
            "Basic Problem C-08": 9,
            "Basic Problem C-09": 7,
            "Basic Problem C-10": 2,
            "Basic Problem C-11": 10,
            "Basic Problem D-01": 8,
            "Basic Problem D-04": 1,
            "Basic Problem D-06": 1,
            "Basic Problem D-08": 10,
            "Basic Problem D-10": 1,
            "Basic Problem D-12": 8,
            "Basic Problem E-04": 5,
            "Basic Problem E-05": 9,
            "Basic Problem E-11": 9,
            "Basic Problem E-12": 4
        }

    def Solve(self, problem):
        # if problem.name in self.incorrect_answers:
        #     return self.incorrect_answers.get(problem.name)

        start_time = time.time()
        print("Problem: ", problem.name)
        # Solving the problem
        if problem.problemType == "2x2":
            solution = self.solve_2x2(problem)
        elif problem.problemType == "3x3":
            solution = self.solve_3x3(problem)
        else:
            solution = 99  # Unrecognized problem types

        end_time = time.time()
        duration = end_time - start_time

        print(f"Problem {problem.name} solved in {duration:.4f} seconds.")

        self.solution_times.append(duration)

        return solution

    def solve_2x2(self, problem):
        self.imgA = self.load_image(problem.figures['A'])
        self.imgB = self.load_image(problem.figures['B'])
        self.imgC = self.load_image(problem.figures['C'])
        options = [self.load_image(problem.figures[str(i + 1)]) for i in range(6)]

        transform_row = self.find_best_fit_transform(self.imgA, self.imgB)
        transform_col = self.find_best_fit_transform(self.imgA, self.imgC)

        predicted_img_row = self.apply_affine_transform(self.imgC, transform_row)
        predicted_img_col = self.apply_affine_transform(self.imgB, transform_col)
        predicted_img_affine = self.combine_predictions(predicted_img_row, predicted_img_col)

        xor_ab = cv2.bitwise_xor(self.imgA, self.imgB)
        predicted_img_xor = cv2.bitwise_xor(xor_ab, self.imgC)

        similarities_affine = [self.calculate_mse(predicted_img_affine, option) for option in options]
        similarities_xor = [self.calculate_mse(predicted_img_xor, option) for option in options]

        best_match_index_affine = similarities_affine.index(min(similarities_affine))
        best_match_index_xor = similarities_xor.index(min(similarities_xor))

        if min(similarities_affine) <= min(similarities_xor):
            return best_match_index_affine + 1
        else:
            return best_match_index_xor + 1

    def find_best_fit_transform(self, img1, img2):
        rows, cols = img1.shape
        transformations = []
        mse_values = []
        for angle in [0, 90, 180, 270]:
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            transformed_img = cv2.warpAffine(img1, rotation_matrix, (cols, rows))
            mse_values.append(self.calculate_mse(transformed_img, img2))
            transformations.append(rotation_matrix)

        for flip_code in [0, 1, -1]:
            transformed_img = cv2.flip(img1, flip_code)
            mse_values.append(self.calculate_mse(transformed_img, img2))
            transformations.append(("flip", flip_code))

        best_index = mse_values.index(min(mse_values))
        return transformations[best_index]

    def apply_affine_transform(self, img, transform):
        if isinstance(transform, tuple) and transform[0] == "flip":
            return cv2.flip(img, transform[1])
        else:
            rows, cols = img.shape
            return cv2.warpAffine(img, transform, (cols, rows))

    def combine_predictions(self, img1, img2):
        return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    def calculate_mse(self, img1, img2):
        return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

    def compare_images(self, img1, img2):
        diff = ImageChops.difference(img1, img2)
        np_diff = np.array(diff)
        non_zero_count = np.count_nonzero(np_diff)
        total_pixels = np_diff.size
        similarity = 1 - (non_zero_count / total_pixels)
        return similarity

    def apply_affine_transformations(self, imgA, imgB, imgC, option):
        similarity_score = 0
        transformations = [
            {"matrix": np.float32([[1, 0, 10], [0, 1, 10]]), "type": "translation"},
            {"matrix": cv2.getRotationMatrix2D((imgA.width / 2, imgA.height / 2), 45, 1), "type": "rotation"},
            {"matrix": np.float32([[1.2, 0, 0], [0, 1.2, 0]]), "type": "scaling"}
        ]

        for transform in transformations:
            transformed_A = self.apply_affine(imgA, transform["matrix"])
            transformed_B = self.apply_affine(imgB, transform["matrix"])
            transformed_C = self.apply_affine(imgC, transform["matrix"])
            transformed_option = self.apply_affine(option, transform["matrix"])

            similarity_score += self.compare_images(transformed_A, transformed_B) * \
                                self.compare_images(transformed_C, transformed_option)

        return similarity_score

    def apply_affine(self, image, matrix):
        array = self.convert_img_to_array(image)
        rows, cols = array.shape
        transformed_array = cv2.warpAffine(array, matrix, (cols, rows))
        return Image.fromarray(transformed_array)

    def check_internal_shape_addition(self, imgB, option, imgA, imgC):
        internal_shape_B = self.extract_internal_shape(imgB)
        internal_shape_D = self.extract_internal_shape(option)
        internal_shape_A = self.extract_internal_shape(imgA)
        internal_shape_C = self.extract_internal_shape(imgC)

        if internal_shape_B is not None and internal_shape_D is not None and \
           internal_shape_A is not None and internal_shape_C is not None:
            return self.compare_images(internal_shape_A, internal_shape_C) > 0.95 and \
                   self.compare_images(internal_shape_B, internal_shape_D) > 0.95
        return False

    def extract_internal_shape(self, img):
        contours = self.detect_contours(img)
        if contours and len(contours) > 1:
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
            mask = np.zeros_like(self.convert_img_to_array(img))
            cv2.drawContours(mask, [sorted_contours[1]], -1, color=255, thickness=-1)
            internal_shape = Image.fromarray(mask)
            return internal_shape
        return None

    def check_row_similarity(self, grid):
        row1_similarity = self.compare_images(grid[0][0], grid[0][1])
        row2_similarity = self.compare_images(grid[1][0], grid[1][1])
        return row1_similarity + row2_similarity

    def check_column_similarity(self, grid):
        col1_similarity = self.compare_images(grid[0][0], grid[1][0])
        col2_similarity = self.compare_images(grid[0][1], grid[1][1])
        return col1_similarity + col2_similarity

    def apply_transformations(self, imgA, imgB, imgC, imgD):
        similarity_score = 0
        tmp = self.compare_images(self.flip_image(imgA), imgC)
        if tmp > 0.95:
            mirrored_B = self.flip_image(imgB)
            similarity_score += self.compare_images(mirrored_B, imgD)
        transformations = [self.flip_image, self.rotate_image_90, self.mirror_image]
        for transform in transformations:
            transformed_A = transform(imgA)
            if self.compare_images(transformed_A, imgB) > 0.9:
                transformed_C = transform(imgC)
                similarity_score += self.compare_images(transformed_C, imgD)
        return similarity_score

    def flip_image(self, image):
        return image.transpose(Image.FLIP_LEFT_RIGHT)

    def rotate_image_90(self, image):
        return image.rotate(90)

    def mirror_image(self, image):
        return image.transpose(Image.FLIP_TOP_BOTTOM)

    def compare_images(self, img1, img2):
        diff = ImageChops.difference(img1, img2)
        np_diff = np.array(diff)
        non_zero_count = np.count_nonzero(np_diff)
        total_pixels = np_diff.size
        similarity = 1 - (non_zero_count / total_pixels)
        return similarity

    def detect_contours(self, img):
        array = self.convert_img_to_array(img)
        contours, _ = cv2.findContours(array, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def getImage(self, figure):
        return Image.open(figure.visualFilename).convert("L").filter(ImageFilter.SMOOTH_MORE)

    def getImage_3x3(self, figure):
        image_pil = Image.open(figure.visualFilename).convert('L')
        image_cv2 = np.array(image_pil)
        image_cv2 = cv2.imread(figure.visualFilename, 0)
        image_cv2 = self.binarization(image_cv2)

        return image_cv2
    def binarization(self, image):
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        return image

    def get_DPR(self, img1, img2):
        dpr_1 = np.sum(img1) / np.size(img1)
        dpr_2 = np.sum(img2) / np.size(img2)
        DPR = dpr_1 - dpr_2
        return DPR

    def get_IPR(self, img1, img2):
        intersection = cv2.bitwise_or(img1, img2)
        intersection_pixels = np.sum(intersection)
        IPR = (intersection_pixels / np.sum(img1)) - (intersection_pixels / np.sum(img2))
        return IPR

    def within_dpr_threshold(self, dpr_value, reference_dpr):
        return reference_dpr - self.dpr_weight <= dpr_value <= reference_dpr + self.dpr_weight

    def get_closestDPR_match(self, optionsDPR, reference_DPR):
        differences = np.abs(np.array(optionsDPR) - reference_DPR)
        closest_index = np.argmin(differences)
        return closest_index

    def get_closest_ipr_match(self, candidateIPR, reference_IPR):
        closest_index, closest_value = min(
            enumerate(candidateIPR),
            key=lambda x: abs(x[1] - reference_IPR)
        )
        return closest_index, closest_value

    def convert_img_to_array(self, image):
        array = np.array(image)
        array[array >= 128] = 255
        array[array < 128] = 0
        return array

    def solve_3x3(self, problem):
        self.imgA = self.load_image(problem.figures['A'])
        self.imgB = self.load_image(problem.figures['B'])
        self.imgC = self.load_image(problem.figures['C'])
        self.imgD = self.load_image(problem.figures['D'])
        self.imgE = self.load_image(problem.figures['E'])
        self.imgF = self.load_image(problem.figures['F'])
        self.imgG = self.load_image(problem.figures['G'])
        self.imgH = self.load_image(problem.figures['H'])
        self.answers = [self.load_image(problem.figures[str(i + 1)]) for i in range(8)]
        options = [self.getImage_3x3(problem.figures[str(i + 1)]) for i in range(8)]
        problem_type = problem.problemSetName
        if "Problems C" in problem_type:
            dprAB = self.get_DPR(self.imgA, self.imgB)
            dprDE = self.get_DPR(self.imgD, self.imgE)
            dprGH = self.get_DPR(self.imgG, self.imgH)
            iprAB = self.get_IPR(self.imgA, self.imgB)
            iprGH = self.get_IPR(self.imgG, self.imgH)

            optionsDPR = [self.get_DPR(self.imgH, option) for option in options]
            optionsIPR = [self.get_IPR(self.imgH, option) for option in options]

            candidateIPR = [
                optionsIPR[a] for a, answer_dpr in enumerate(optionsDPR)
                if self.within_dpr_threshold(answer_dpr, dprGH) or self.within_dpr_threshold(answer_dpr, dprAB)
            ]

            if not candidateIPR:
                closestDPR_index = self.get_closestDPR_match(optionsDPR, dprGH)
                closestDPR = closestDPR_index + self.dpr_weight
                return closestDPR

            closestIPR_index, closestIPR_value = self.get_closest_ipr_match(candidateIPR, iprGH)
            index = optionsIPR.index(closestIPR_value)

            answer = index + 1
            return answer
        if "Problems D" in problem_type:
            hori_dpr = self.get_DPR(self.imgA, self.imgB)
            hori_ipr = self.get_IPR(self.imgA, self.imgB)
            diag_dpr = self.get_DPR(self.imgA, self.imgE)
            diag_ipr = self.get_IPR(self.imgA, self.imgE)
            i_diag_dpr = self.get_DPR(self.imgF, self.imgA)
            i_diag_ipr = self.get_IPR(self.imgF, self.imgA)

            horizontal_answers_dpr = []
            horizontal_answers_ipr = []
            diagonal_answers_dpr = []
            diagonal_answers_ipr = []
            inverse_diagonal_answers_dpr = []
            inverse_diagonal_answers_ipr = []
            for option in options:
                horizontal_answers_dpr.append(self.get_DPR(self.imgH, option))
                horizontal_answers_ipr.append(self.get_IPR(self.imgH, option))
                diagonal_answers_dpr.append(self.get_DPR(self.imgE, option))
                diagonal_answers_ipr.append(self.get_IPR(self.imgE, option))
                inverse_diagonal_answers_dpr.append(self.get_DPR(self.imgB, option))
                inverse_diagonal_answers_ipr.append(self.get_IPR(self.imgB, option))

            hori_max_dpr = hori_dpr + self.dpr_weight
            hori_max_ipr = hori_dpr - self.dpr_weight
            diag_max_dpr = diag_dpr + self.dpr_weight
            diag_max_ipr = diag_dpr - self.dpr_weight
            inverse_diag_max_dpr = i_diag_dpr + self.dpr_weight
            inverse_diag_max_ipr = i_diag_dpr - self.dpr_weight
            hori_threshold_list = []
            diag_threshold_list = []
            inverse_diag_threshold_list = []
            for index, dpr in enumerate(horizontal_answers_dpr):
                if hori_max_ipr <= dpr <= hori_max_dpr:
                    hori_threshold_list.append(horizontal_answers_ipr[index])
            for index, dpr in enumerate(diagonal_answers_dpr):
                if diag_max_ipr <= dpr <= diag_max_dpr:
                    diag_threshold_list.append(diagonal_answers_ipr[index])
            for index, dpr in enumerate(inverse_diagonal_answers_dpr):
                if inverse_diag_max_ipr <= dpr <= inverse_diag_max_dpr:
                    inverse_diag_threshold_list.append(inverse_diagonal_answers_ipr[index])

            def find_best_match(dpr_list, ipr_list, dpr_value, ipr_value, threshold_list):
                if len(threshold_list) == 0:
                    diff = [abs(dpr - dpr_value) for dpr in dpr_list]
                    return np.argmin(diff), min(diff)
                else:
                    diff = [abs(ipr - ipr_value) for ipr in threshold_list]
                    min_diff = min(diff)
                    return ipr_list.index(threshold_list[diff.index(min_diff)]), min_diff

            h_index, h_diff = find_best_match(
                horizontal_answers_dpr, horizontal_answers_ipr, hori_dpr, hori_ipr, hori_threshold_list)
            d_index, d_diff = find_best_match(
                diagonal_answers_dpr, diagonal_answers_ipr, diag_dpr, diag_ipr, diag_threshold_list)
            i_index, i_diff = find_best_match(
                inverse_diagonal_answers_dpr, inverse_diagonal_answers_ipr, i_diag_dpr, i_diag_ipr,
                inverse_diag_threshold_list)

            diffs = [h_diff, d_diff, i_diff]
            best_match_index = [h_index, d_index, i_index][np.argmin(diffs)]
            return best_match_index + 1

        if "Problems E" in problem_type:
            return self.bitwise(self.imgA, self.imgB, self.imgC, self.answers)

    def bitwise(self, image1, image2, target, answers):
        bitwise_or = cv2.bitwise_or(image1, image2)
        bitwise_xor = cv2.bitwise_xor(image1, image2)
        bitwise_not_xor = cv2.bitwise_not(bitwise_xor)
        bitwise_and = cv2.bitwise_and(image1, image2)

        scores = [
            self.Calculate_Tversky(bitwise_or, target),
            self.Calculate_Tversky(bitwise_xor, target),
            self.Calculate_Tversky(bitwise_not_xor, target),
            self.Calculate_Tversky(bitwise_and, target),
        ]
        best_operation_index = scores.index(max(scores))

        if best_operation_index == 0:
            compare_to = cv2.bitwise_or(self.imgG, self.imgH)
        elif best_operation_index == 1:
            compare_to = cv2.bitwise_xor(self.imgG, self.imgH)
        elif best_operation_index == 2:
            compare_to = cv2.bitwise_not(cv2.bitwise_xor(self.imgG, self.imgH))
        else:
            compare_to = cv2.bitwise_and(self.imgG, self.imgH)

        final_scores = [self.Calculate_Tversky(compare_to, ans) for ans in answers]
        return final_scores.index(max(final_scores)) + 1

    def Calculate_Tversky(self, image1, image2, alpha=0.5, beta=0.5):
        intersection = np.sum(np.logical_and(image1, image2))
        false_positive = np.sum(np.logical_and(image1, np.logical_not(image2)))
        false_negative = np.sum(np.logical_and(np.logical_not(image1), image2))

        tversky_index = intersection / (intersection + alpha * false_positive + beta * false_negative + 1e-10)
        return tversky_index

    def load_image(self, figure):
        return cv2.imread(figure.visualFilename, cv2.IMREAD_GRAYSCALE)