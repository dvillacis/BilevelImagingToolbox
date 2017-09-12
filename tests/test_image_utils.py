from bilevel_imaging_toolbox import image_utils
from bilevel_imaging_toolbox import operators

circle_image_path = "../examples/images/circle.png"

circle_image = image_utils.load_image(circle_image_path)
gray_circle_image = image_utils.convert_to_grayscale(circle_image)
print(gray_circle_image.shape)
image_utils.show_image(gray_circle_image,'circle')

lena_image_path = "../examples/images/lena.png"
lena_image = image_utils.load_image(lena_image_path)
gray_lena_image = image_utils.convert_to_grayscale(lena_image)
print(gray_lena_image.shape)
image_utils.show_image(gray_lena_image,'lena')

# Forward Finite differences plotting test
small_circle_image_path = "../examples/images/circle_24.png"
small_circle = image_utils.load_image(small_circle_image_path)
small_circle = image_utils.convert_to_grayscale(small_circle)
op = operators.make_finite_differences_operator(small_circle.shape,'fn',1)
grad = op.val(small_circle)
image_list = [small_circle,grad[:,:,0],grad[:,:,1]]
image_names = ["original","gradx","grady"]
image_utils.show_collection(image_list,image_names)

# Backward Finite differences plotting test
small_circle_image_path = "../examples/images/circle_24.png"
small_circle = image_utils.load_image(small_circle_image_path)
small_circle = image_utils.convert_to_grayscale(small_circle)
op = operators.make_finite_differences_operator(small_circle.shape,'bn',1)
grad = op.val(small_circle)
image_list = [small_circle,grad[:,:,0],grad[:,:,1]]
image_names = ["original","gradx","grady"]
image_utils.show_collection(image_list,image_names)

# Centered Finite differences plotting test
small_circle_image_path = "../examples/images/circle_24.png"
small_circle = image_utils.load_image(small_circle_image_path)
small_circle = image_utils.convert_to_grayscale(small_circle)
op = operators.make_finite_differences_operator(small_circle.shape,'cn',1)
grad = op.val(small_circle)
image_list = [small_circle,grad[:,:,0],grad[:,:,1]]
image_names = ["original","gradx","grady"]
image_utils.show_collection(image_list,image_names)
