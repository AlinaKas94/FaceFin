#include "FaceEngine.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <future> // Для паралельного батчингу

// --- ФІКС КРАШУ ONNX RUNTIME (ПОСТІЙНИЙ ВКАЗІВНИК) ---
static Ort::Env &getOrtEnv()
{
    static Ort::Env *env = nullptr;
    if (!env)
    {
        env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "FaceFinEngine");
    }
    return *env;
}

FaceEngine::FaceEngine(const std::wstring &detModelPath, const std::wstring &recModelPath, bool useGPU)
{
    try
    {
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        if (useGPU)
        {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            sessionOptions.AppendExecutionProvider_CUDA(cuda_options);
        }

        detSession = std::make_unique<Ort::Session>(getOrtEnv(), detModelPath.c_str(), sessionOptions);
        recSession = std::make_unique<Ort::Session>(getOrtEnv(), recModelPath.c_str(), sessionOptions);

        modelsLoaded = true;
    }
    catch (const std::exception &e)
    {
        lastError = e.what();
        modelsLoaded = false;
    }
}

FaceEngine::~FaceEngine() {}

bool FaceEngine::isLoaded() const { return modelsLoaded; }
std::string FaceEngine::getLastError() const { return lastError; }

void FaceEngine::nms(std::vector<FaceInfo> &faces, float nmsThreshold)
{
    if (faces.empty())
        return;
    std::sort(faces.begin(), faces.end(), [](const FaceInfo &a, const FaceInfo &b)
              { return a.score > b.score; });
    std::vector<FaceInfo> result;
    std::vector<bool> suppressed(faces.size(), false);

    for (size_t i = 0; i < faces.size(); i++)
    {
        if (suppressed[i])
            continue;
        result.push_back(faces[i]);
        for (size_t j = i + 1; j < faces.size(); j++)
        {
            if (suppressed[j])
                continue;
            cv::Rect2f inter = faces[i].bbox & faces[j].bbox;
            float interArea = inter.area();
            float unionArea = faces[i].bbox.area() + faces[j].bbox.area() - interArea;
            if (unionArea > 0.0f && (interArea / unionArea) > nmsThreshold)
                suppressed[j] = true;
        }
    }
    faces = result;
}

std::vector<FaceInfo> FaceEngine::detectFaces(const cv::Mat &image, float scoreThreshold) {
    std::vector<FaceInfo> faces;
    if (!modelsLoaded || image.empty()) return faces;

    try {
        // --- НОВИЙ АЛГОРИТМ: Letterbox (Збереження пропорцій) ---
        int targetSize = 640;
        float scale = std::min((float)targetSize / image.cols, (float)targetSize / image.rows);
        int newW = std::round(image.cols * scale);
        int newH = std::round(image.rows * scale);

        cv::Mat resizedImage;
        cv::resize(image, resizedImage, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

        // Створюємо нейтральний сірий фон (127.5, 127.5, 127.5) для порожніх зон
        cv::Mat canvas(targetSize, targetSize, CV_8UC3, cv::Scalar(127.5, 127.5, 127.5));
        
        // Вставляємо масштабоване фото у верхній лівий кут
        resizedImage.copyTo(canvas(cv::Rect(0, 0, newW, newH)));

        cv::Mat blob;
        // УВАГА: swapRB = true (Конвертуємо BGR в RGB для правильного сприйняття шкіри)
        cv::dnn::blobFromImage(canvas, blob, 1.0 / 128.0, cv::Size(targetSize, targetSize), cv::Scalar(127.5, 127.5, 127.5), true, false);

        std::vector<int64_t> inputDims = {1, 3, targetSize, targetSize};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, (float *)blob.data, blob.total(), inputDims.data(), inputDims.size());

        Ort::AllocatedStringPtr inputName = detSession->GetInputNameAllocated(0, allocator);
        const char *inputNames[] = {inputName.get()};

        size_t numOutputs = detSession->GetOutputCount();
        std::vector<Ort::AllocatedStringPtr> allocatedOutputNames;
        allocatedOutputNames.reserve(numOutputs);
        std::vector<const char *> outputNames;
        outputNames.reserve(numOutputs);

        for (size_t i = 0; i < numOutputs; i++) {
            allocatedOutputNames.push_back(detSession->GetOutputNameAllocated(i, allocator));
            outputNames.push_back(allocatedOutputNames.back().get());
        }

        auto outputTensors = detSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames.data(), outputNames.size());

        if (numOutputs != 6 && numOutputs != 9) return faces;

        int fmc = 3;
        std::vector<const float *> scores_ptrs(fmc);
        std::vector<const float *> bboxes_ptrs(fmc);
        std::vector<const float *> kps_ptrs(fmc, nullptr);

        for (int i = 0; i < fmc; ++i) {
            scores_ptrs[i] = outputTensors[i].GetTensorMutableData<float>();
            bboxes_ptrs[i] = outputTensors[i + fmc].GetTensorMutableData<float>();
            if (numOutputs == 9) kps_ptrs[i] = outputTensors[i + fmc * 2].GetTensorMutableData<float>();
        }

        int strides[] = {8, 16, 32};
        for (int i = 0; i < fmc; ++i) {
            const float *score_ptr = scores_ptrs[i];
            const float *bbox_ptr = bboxes_ptrs[i];
            const float *kps_ptr = kps_ptrs[i];
            int stride = strides[i];
            int feature_map_w = targetSize / stride;
            int feature_map_h = targetSize / stride;
            int idx = 0;

            for (int y = 0; y < feature_map_h; ++y) {
                for (int x = 0; x < feature_map_w; ++x) {
                    for (int a = 0; a < 2; ++a) {
                        float score = score_ptr[idx];
                        if (score >= scoreThreshold) {
                            float anchor_x = x * stride;
                            float anchor_y = y * stride;
                            FaceInfo face;
                            face.score = score;
                            
                            // НОВА МАТЕМАТИКА: Ділимо координати на масштаб, щоб повернути їх на оригінальне фото
                            float x1 = (anchor_x - bbox_ptr[idx * 4 + 0] * stride) / scale;
                            float y1 = (anchor_y - bbox_ptr[idx * 4 + 1] * stride) / scale;
                            float x2 = (anchor_x + bbox_ptr[idx * 4 + 2] * stride) / scale;
                            float y2 = (anchor_y + bbox_ptr[idx * 4 + 3] * stride) / scale;
                            face.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

                            if (kps_ptr != nullptr) {
                                for (int k = 0; k < 5; ++k) {
                                    float kx = (anchor_x + kps_ptr[idx * 10 + k * 2 + 0] * stride) / scale;
                                    float ky = (anchor_y + kps_ptr[idx * 10 + k * 2 + 1] * stride) / scale;
                                    face.kps.push_back(cv::Point2f(kx, ky));
                                }
                            }
                            faces.push_back(face);
                        }
                        idx++;
                    }
                }
            }
        }
        nms(faces, 0.4f);
    } catch (const std::exception &e) {
        lastError = std::string("Детекція Помилка: ") + e.what();
    }
    return faces;
}

cv::Mat FaceEngine::alignFace(const cv::Mat &image, const FaceInfo &face) const
{
    if (face.kps.size() != 5)
        return cv::Mat();
    for (const auto &pt : face.kps)
    {
        if (std::isnan(pt.x) || std::isnan(pt.y))
            return cv::Mat();
    }
    std::vector<cv::Point2f> dst_pts = {
        cv::Point2f(38.2946f, 51.6963f), cv::Point2f(73.5318f, 51.5014f),
        cv::Point2f(56.0252f, 71.7366f), cv::Point2f(41.5493f, 92.3655f),
        cv::Point2f(70.7299f, 92.2041f)};
    cv::Mat M = cv::estimateAffinePartial2D(face.kps, dst_pts);
    if (M.empty())
        return cv::Mat();
    cv::Mat alignedFace;
    cv::warpAffine(image, alignedFace, M, cv::Size(112, 112), cv::INTER_LINEAR);
    return alignedFace;
}

bool FaceEngine::extractEmbedding(const cv::Mat &image, FaceInfo &face)
{
    if (!modelsLoaded || image.empty())
        return false;
    try
    {
        cv::Mat alignedFace = alignFace(image, face);
        if (alignedFace.empty())
        {
            lastError = "Не вдалося вирівняти обличчя.";
            return false;
        }

        cv::Mat blob;
        cv::dnn::blobFromImage(alignedFace, blob, 1.0 / 127.5, cv::Size(112, 112), cv::Scalar(127.5, 127.5, 127.5), true, false);

        std::vector<int64_t> recInputDims = {1, 3, 112, 112};
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, (float *)blob.data, blob.total(), recInputDims.data(), recInputDims.size());

        Ort::AllocatedStringPtr inputName = recSession->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr outputName = recSession->GetOutputNameAllocated(0, allocator);
        const char *inputNames[] = {inputName.get()};
        const char *outputNames[] = {outputName.get()};

        auto outputTensors = recSession->Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

        if (outputTensors.empty())
            return false;

        const float *embeddingData = outputTensors.front().GetTensorMutableData<float>();
        face.embedding.assign(embeddingData, embeddingData + 512);

        float norm = 0.0f;
        for (float val : face.embedding)
            norm += val * val;
        norm = std::sqrt(norm);
        if (norm > 0.0001f)
        {
            for (float &val : face.embedding)
                val /= norm;
        }
        return true;
    }
    catch (const std::exception &e)
    {
        lastError = std::string("Векторизація Помилка: ") + e.what();
        return false;
    }
}

// --- НОВИЙ СУПЕРШВИДКИЙ ПАРАЛЕЛЬНИЙ БАТЧИНГ (C++ ASYNC) ---
bool FaceEngine::extractEmbeddingsBatch(const std::vector<cv::Mat> &alignedFaces, std::vector<std::vector<float>> &outputEmbeddings)
{
    if (alignedFaces.empty() || !isLoaded())
        return false;

    int64_t batchSize = static_cast<int64_t>(alignedFaces.size());
    outputEmbeddings.resize(batchSize);

    try
    {
        Ort::AllocatedStringPtr inputName = recSession->GetInputNameAllocated(0, allocator);
        Ort::AllocatedStringPtr outputName = recSession->GetOutputNameAllocated(0, allocator);

        // Зберігаємо імена у std::string, щоб безпечно передавати їх у паралельні потоки
        std::string inNameStr = inputName.get();
        std::string outNameStr = outputName.get();

        std::vector<std::future<void>> futures;

        // Запускаємо паралельну обробку для кожного обличчя в батчі
        for (int i = 0; i < batchSize; ++i)
        {
            futures.push_back(std::async(std::launch::async, [this, &alignedFaces, i, &outputEmbeddings, inNameStr, outNameStr]()
                                         {
                // Кожен потік створює власне локальне середовище пам'яті для безпеки
                auto localMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                
                cv::Mat singleBlob;
                cv::dnn::blobFromImage(alignedFaces[i], singleBlob, 1.0 / 127.5, cv::Size(112, 112), cv::Scalar(127.5, 127.5, 127.5), true, false);
                
                std::vector<int64_t> singleShape = {1, 3, 112, 112};
                Ort::Value singleInput = Ort::Value::CreateTensor<float>(
                    localMemoryInfo, reinterpret_cast<float *>(singleBlob.data), singleBlob.total(), singleShape.data(), singleShape.size());
                
                const char* localInputNames[] = {inNameStr.c_str()};
                const char* localOutputNames[] = {outNameStr.c_str()};

                // Thread-Safe виклик нейромережі
                auto singleOut = recSession->Run(Ort::RunOptions{nullptr}, localInputNames, &singleInput, 1, localOutputNames, 1);
                float *outData = singleOut[0].GetTensorMutableData<float>();
                
                // Записуємо результат безпосередньо у виділений для цього потоку індекс (Thread-Safe)
                outputEmbeddings[i].resize(512);
                std::memcpy(outputEmbeddings[i].data(), outData, 512 * sizeof(float));
                
                // Нормалізація вектора (L2)
                float norm = 0.0f;
                for (float v : outputEmbeddings[i]) norm += v * v;
                norm = std::sqrt(norm);
                if (norm > 0.0001f) {
                    for (float &v : outputEmbeddings[i]) v /= norm;
                } }));
        }

        // Чекаємо, поки всі потоки завершать векторизацію (Синхронізація)
        for (auto &f : futures)
        {
            f.get();
        }

        return true;
    }
    catch (const std::exception &e)
    {
        lastError = std::string("Батчинг Помилка: ") + e.what();
        return false;
    }
}

float FaceEngine::compareEmbeddings(const std::vector<float> &emb1, const std::vector<float> &emb2) const
{
    if (emb1.size() != 512 || emb2.size() != 512)
        return 0.0f;
    float dotProduct = 0.0f;
    for (size_t i = 0; i < 512; ++i)
        dotProduct += emb1[i] * emb2[i];
    return dotProduct;
}