#include <opencv2/opencv.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>
#include <chrono>
#include <cassert>

// Include the header where xyxy2xywh is declared
#include "improc.h"

bool test_1(){

    double m[1][4] = {{100, 100, 200, 200}};
    cv::Mat input_cpu(1, 4, CV_64F, m);
    input_cpu.inv();

    // Upload the image to GPU
    cv::cuda::GpuMat d_box;
    d_box.upload(input_cpu);

    // Measure the time taken to rotate the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output = improc::xyxy2xywh(d_box);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to convert xyxy2xywh: " << duration.count() << " seconds." << std::endl;

    // Download the result back to CPU and display
    cv::Mat result;
    output.download(result);

    // Define the reference output
    double ref_data[1][4] = {{150, 150, 100, 100}};
    cv::Mat reference_output = cv::Mat(1, 4, CV_64F, ref_data);
    reference_output.inv();

    // Compare the result with the reference output
    cv::Mat diff;
    cv::absdiff(result, reference_output, diff);
    double max_diff = cv::norm(diff, cv::NORM_INF);

    // Assert that the maximum difference is within an acceptable range
    try{
        assert(max_diff < 1e-6);
        std::cout << "Test 1 assed. The predicted output matches the reference output." << std::endl;
        return true;
    }
    catch (const std::exception& e){
        std::cout << "The predicted output differs from the reference output!" << std::endl;
        return false;
    }
}

bool test_2(){

    double m[1][4] = {{150, 150, 100, 100}};
    cv::Mat input_cpu(1, 4, CV_64F, m);
    input_cpu.inv();

    // Upload the image to GPU
    cv::cuda::GpuMat d_box;
    d_box.upload(input_cpu);

    std::cout << "Input: " << input_cpu << std::endl;

    // Measure the time taken to rotate the image
    auto start = std::chrono::high_resolution_clock::now();
    cv::cuda::GpuMat output = improc::xywh2xyxy(d_box);
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to convert xywh2xyxy: " << duration.count() << " seconds." << std::endl;

    // Download the result back to CPU and display
    cv::Mat result;
    output.download(result);

    // Define the reference output
    double ref_data[1][4] = {{100, 100, 200, 200}};
    cv::Mat reference_output = cv::Mat(1, 4, CV_64F, ref_data);
    reference_output.inv();

    std::cout << "Reference output: " << reference_output << std::endl;
    std::cout << "Result: " << result << std::endl;

    // Compare the result with the reference output
    cv::Mat diff;
    cv::absdiff(result, reference_output, diff);
    double max_diff = cv::norm(diff, cv::NORM_INF);

    // Assert that the maximum difference is within an acceptable range
    try{
        assert(max_diff < 1e-6);
        std::cout << "Test 2 assed. The predicted output matches the reference output." << std::endl;
        return true;
    }
    catch (const std::exception& e){
        std::cout << "The predicted output differs from the reference output!" << std::endl;
        return false;
    }
}

int main() {
    test_1();
    test_2();
    return 0;
}