#pragma once
#include <string>
#include <vector>
#include <cstdint> // Додано для int64_t
#include <sqlite3.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>

struct SearchResult
{
    int64_t id; // <--- ДОДАНО: Унікальний номер (rowid) з бази даних
    std::string filePath;
    int pageNum;
    float similarity;
    std::string name;
    std::string cachePath;
    std::vector<std::string> duplicatePaths;
};

class Database
{
public:
    Database(const std::string &dbPath);
    ~Database();

    bool isReady() const;
    std::string getLastError() const;
    int getRecordCount() const;

    // Перевірка: чи був цей файл уже проіндексований раніше
    bool isFileIndexed(const std::string &filePath);

    bool enroll(const std::string &filePath, int pageNum, const std::vector<float> &embedding, const std::string &name = "", const std::string &cachePath = "");
    std::vector<SearchResult> search(const std::vector<float> &queryEmbedding, float threshold = 0.45f, int maxResults = 60);

    // <--- ЗМІНЕНО: Тепер приймає унікальний id замість filePath
    bool updateName(int64_t id, const std::string &newName);
    void clearDatabase();

    bool saveIndex();

    std::vector<float> getEmbeddingById(int64_t id);

private:
    sqlite3 *db = nullptr;
    bool dbReady = false;
    std::string lastError;

    std::string m_indexPath;
    faiss::IndexFlatIP *baseIndex = nullptr;
    faiss::IndexIDMap *idMapIndex = nullptr;

    bool loadFaissIndex();
    bool getFileInfoById(int64_t id, std::string &filePath, int &pageNum, std::string &name, std::string &cachePath);
};