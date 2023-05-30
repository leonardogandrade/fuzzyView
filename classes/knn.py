"""
    KNN color clustering module
"""


from sklearn.cluster import KMeans
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
from PIL import ImageColor

class KnnColors:
    def __init__(self, image_path, number_cluster) -> None:
        self.image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.number_cluster = number_cluster
        self.modified_image = self.image.reshape(self.image.shape[0] * self.image.shape[1], 3)
        self.clf = KMeans(n_clusters = number_cluster, n_init='auto')
        self.labels = self.clf.fit_predict(self.modified_image)
    
    def RGB2HEX(self, color):
        return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

    def get_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    @staticmethod
    def compare_colors(color_list: list, base_color: tuple, delta: int):
        base_color_rgb = ImageColor.getcolor(base_color, "RGB")
        result = False
        
        for color in color_list:
            color_rgb = ImageColor.getcolor(color, "RGB")
            diff = deltaE_cie76(base_color_rgb, color_rgb)
            if diff <= delta:
                result = True
                print(color)
        
        return result

    def get_colors(self):
        counts = Counter(self.labels)

        center_colors = self.clf.cluster_centers_
        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]
        # if (show_chart):
        #     plt.figure(figsize = (8, 6))
        #     plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
            
        return hex_colors


    def color_compare(color_list: list, base_color: tuple, delta: int):
        base_color_rgb = ImageColor.getcolor(base_color, "RGB")
        result = False
        
        for color in color_list:
            color_rgb = ImageColor.getcolor(color, "RGB")
            diff = deltaE_cie76(base_color_rgb, color_rgb)
            if diff <= delta:
                result = True
        
        return result

    def skip_colors(self, skip_colors, color_fuzzy_distance: int = 50):
        counts = Counter(self.labels)
        center_colors = self.clf.cluster_centers_
        
        # We get ordered colors by iterating through the keys
        ordered_colors = [center_colors[i] for i in counts.keys()]
        
        ordered_colors_copy = ordered_colors.copy()
        
        for idx, val in enumerate(ordered_colors_copy):
            for color in skip_colors.keys():
                if deltaE_cie76(val, skip_colors[color]) < color_fuzzy_distance:
                    del counts[idx]
        
        hex_colors = [self.RGB2HEX(ordered_colors[i]) for i in counts.keys()]
        rgb_colors = [ordered_colors[i] for i in counts.keys()]
            
        return hex_colors


# skip_colors_list = {
#     'grey': [191, 191, 191],
#     'white': [255, 255, 255]
# }

# img_path = 'classes/batom_2.jpg'
# clusters = 3
# knn = KnnColors(img_path, clusters)

# result = knn.get_colors()
# print(result)

# result_skip_colors = knn.skip_colors(img_path, skip_colors=skip_colors_list, color_fuzzy_distance= 60)
# print(result_skip_colors)

############# compare colors

# color_list = ["#0202f8", "#fa0304", "#02fd04"]
# delta = 200
# base_color = "#6161A6" 
# result = KnnColors.compare_colors(color_list, base_color, delta)
# print(result)
