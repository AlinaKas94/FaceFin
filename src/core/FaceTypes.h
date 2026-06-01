#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// Структура, що описує одне знайдене обличчя
struct FaceInfo
{
    cv::Rect2f bbox;              // Рамка навколо обличчя (x, y, ширина, висота)
    float score;                  // Впевненість нейромережі (наприклад, 0.98)
    std::vector<cv::Point2f> kps; // 5 ключових точок (очі, ніс, рот)
    std::vector<float> embedding; // Майбутній 512-мірний вектор (для розпізнавання)
};