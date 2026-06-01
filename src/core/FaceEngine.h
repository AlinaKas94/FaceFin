#pragma once
#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

struct FaceInfo
{
    cv::Rect2f bbox;
    float score;
    std::vector<cv::Point2f> kps;
    std::vector<float> embedding;
};

class FaceEngine
{
public:
    FaceEngine(const std::wstring &detModelPath, const std::wstring &recModelPath, bool useGPU = false);
    ~FaceEngine();

    bool isLoaded() const;
    std::string getLastError() const;

    std::vector<FaceInfo> detectFaces(const cv::Mat &image, float scoreThreshold = 0.5f);
    cv::Mat alignFace(const cv::Mat &image, const FaceInfo &face) const;

    bool extractEmbedding(const cv::Mat &image, FaceInfo &face);
    bool extractEmbeddingsBatch(const std::vector<cv::Mat> &alignedFaces, std::vector<std::vector<float>> &outputEmbeddings);
    float compareEmbeddings(const std::vector<float> &emb1, const std::vector<float> &emb2) const;

private:
    // ВИДАЛЕНО Ort::Env env; (Тепер воно глобальне у .cpp)
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;

    std::unique_ptr<Ort::Session> detSession;
    std::unique_ptr<Ort::Session> recSession;

    bool modelsLoaded = false;
    std::string lastError;

    void nms(std::vector<FaceInfo> &faces, float nmsThreshold);
};