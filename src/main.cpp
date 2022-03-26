#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <iostream>
#include <map>
#include <opencv2/imgproc.hpp>

using namespace cv;

Mat gaussianFourierKernel(const int width, const int height, const double sigma) {
    Mat kernel(height, width, CV_64F, Scalar(0.0));

    double sum = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int x = j - (width / 2);
            int y = i - (height / 2);
            double value = 1. / (2 * M_PI * sigma * sigma) * exp(-(x * x + y * y) / (2 * sigma * sigma));
            kernel.at<double>(i, j) = value;
            sum += value;
        }
    }
    kernel /= sum;
    return kernel;
}

void f(const Mat &src, Mat &dst, const Mat &depth, std::map<int, Mat> &kernels, int i, int j, unsigned char threshold) {
    double r = 0;
    double g = 0;
    double b = 0;

    // Get the pixel depth
    auto deepness = depth.at<unsigned char>(i, j);

    // Blurring desired area
    if (deepness < threshold) {
        auto blurStrength = (double) (threshold - deepness) / (double) threshold;
        if (kernels.find(deepness) == kernels.end()) {
            kernels[deepness] = gaussianFourierKernel(25, 25, blurStrength * 20);
        }
        auto kernel = kernels[deepness];

        int size = 25;

        //auto ratio = (double) deepness / (double) threshold;

        for (int k = 0; k < size; k++) {
            for (int l = 0; l < size; l++) {
                int x = j + l - (size / 2);
                int y = i + k - (size / 2);

                // If neighbor pixel is not in the desired area,
                // applying a symmetry to get back in the desired area
                if (depth.at<unsigned char>(y, x) > deepness + 10) {
                    y = i + (i - y);
                    x = j + (j - x);
                }

                if (x < 0) x *= -1;
                if (y < 0) y *= -1;
                if (x >= src.cols) x = src.cols - (x - src.cols);
                if (y >= src.rows) y = src.rows - (y - src.rows);

                //std::cout << k << " " << l << " " << x << " " << y << std::endl;
                r += src.at<Vec3b>(y, x)[2] * kernel.at<double>(k, l);
                g += src.at<Vec3b>(y, x)[1] * kernel.at<double>(k, l);
                b += src.at<Vec3b>(y, x)[0] * kernel.at<double>(k, l);
            }
        }

        dst.at<Vec3b>(i, j)[0] = (unsigned char) b;
        dst.at<Vec3b>(i, j)[1] = (unsigned char) g;
        dst.at<Vec3b>(i, j)[2] = (unsigned char) r;
    } else {
        dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
    }
}

int main(int ac, char **av) {
    if (ac != 5) {
        std::cerr << "Usage: " << av[0] << " <image> <depth> <threshold> <output>" << std::endl;
        return 1;
    }
    Mat image = imread(av[1]);
    Mat depth = imread(av[2], IMREAD_GRAYSCALE);
    if (image.empty() || depth.empty()) {
        std::cerr << "Could not open or find the image" << std::endl;
        return 1;
    }


    int kernelSize = 25;
    //Mat kernel(kernelSize, kernelSize, CV_32F, Scalar(1. / (kernelSize * kernelSize)));
    Mat kernel = gaussianFourierKernel(kernelSize, kernelSize, 10);
    std::map<int, Mat> kernels;
    //Mat kernel = getGaussianKernel(kernelSize, 10, CV_64F);

    Mat blurred;
    image.convertTo(blurred, -1);

    int threshold = (int) strtol(av[3], nullptr, 10);

    // Blurring the image on every depth
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            f(image, blurred, depth, kernels, i, j, threshold);
        }
    }
    // Blurring globally the image
    GaussianBlur(blurred, blurred, Size(kernelSize, kernelSize), 3, 3);
    // Putting back the original non blurred area on the blurred image
    for (int i = 0; i < depth.rows; i++) {
        for (int j = 0; j < depth.cols; j++) {
            if (threshold < depth.at<unsigned char>(i, j)) {
                blurred.at<Vec3b>(i, j) = image.at<Vec3b>(i, j);
            }
        }
    }

    // Display everything
    namedWindow("Input image");
    namedWindow("Output image");
    moveWindow("Input image", 50, 50);
    moveWindow("Output image", 600, 50);
    imshow("Input image", image);
    imshow("Output image", blurred);
    waitKey();

    imwrite(av[4], blurred);
    return 0;
}