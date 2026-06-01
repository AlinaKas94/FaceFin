#include "Database.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <faiss/index_io.h> // 👇 ОБОВ'ЯЗКОВО: додаємо заголовок для вводу/виводу FAISS
#include <cstdio>           // Для std::remove (видалення файлу індексу)

using json = nlohmann::json;

Database::Database(const std::string &dbPath)
{
    // Запам'ятовуємо шлях до бінарного індексу поруч із базою SQLite
    m_indexPath = dbPath + ".index";

    if (sqlite3_open(dbPath.c_str(), &db) == SQLITE_OK)
    {
        const char *sqlCreateTable =
            "CREATE TABLE IF NOT EXISTS scanned_files ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "file_path TEXT NOT NULL, "
            "page_num INTEGER NOT NULL, "
            "embedding TEXT NOT NULL, "
            "name TEXT DEFAULT '', "
            "cache_path TEXT DEFAULT '');";

        sqlite3_exec(db, sqlCreateTable, nullptr, nullptr, nullptr);

        sqlite3_exec(db, "CREATE INDEX IF NOT EXISTS idx_file_path ON scanned_files(file_path);", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "ALTER TABLE scanned_files ADD COLUMN name TEXT DEFAULT '';", nullptr, nullptr, nullptr);
        sqlite3_exec(db, "ALTER TABLE scanned_files ADD COLUMN cache_path TEXT DEFAULT '';", nullptr, nullptr, nullptr);
    }
    else
    {
        lastError = "Не вдалося відкрити БД";
        return;
    }

    // 🔴 ЗВЕРНИ УВАГУ: Виділення пам'яті new faiss... перенесено всередину loadFaissIndex()
    if (loadFaissIndex())
        dbReady = true;
}

Database::~Database()
{
    saveIndex(); // 🔥 Автоматично зберігаємо свіжий індекс на диск при закритті програми

    if (idMapIndex)
        delete idMapIndex; // Оскільки own_fields = true, він сам видалить і внутрішній baseIndex

    if (db)
        sqlite3_close(db);
}

bool Database::isReady() const { return dbReady; }
std::string Database::getLastError() const { return lastError; }
int Database::getRecordCount() const { return idMapIndex ? idMapIndex->ntotal : 0; }

bool Database::isFileIndexed(const std::string &filePath)
{
    if (!dbReady)
        return false;
    const char *sql = "SELECT 1 FROM scanned_files WHERE file_path = ? LIMIT 1;";
    sqlite3_stmt *stmt;
    bool exists = false;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK)
    {
        sqlite3_bind_text(stmt, 1, filePath.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            exists = true;
        }
    }
    sqlite3_finalize(stmt);
    return exists;
}

void Database::clearDatabase()
{
    if (!dbReady)
        return;
    sqlite3_exec(db, "DELETE FROM scanned_files;", nullptr, nullptr, nullptr);

    // Видаляємо бінарний файл індексу з диска, бо база тепер порожня
    std::remove(m_indexPath.c_str());

    if (idMapIndex)
        delete idMapIndex;

    baseIndex = new faiss::IndexFlatIP(512);
    idMapIndex = new faiss::IndexIDMap(baseIndex);
    idMapIndex->own_fields = true; // Передаємо керування пам'яттю контейнеру
}

bool Database::loadFaissIndex()
{
    // 1. СТРАТЕГІЯ ШВИДКОГО ЗАПУСКУ: Шукаємо вже готовий бінарний зліпок FAISS
    if (FILE *f = fopen(m_indexPath.c_str(), "rb"))
    {
        fclose(f);
        try
        {
            faiss::Index *idx = faiss::read_index(m_indexPath.c_str());
            if (idx)
            {
                idMapIndex = dynamic_cast<faiss::IndexIDMap *>(idx);
                if (idMapIndex)
                {
                    // Відновлюємо покажчик на базовий індекс всередині мапи
                    baseIndex = dynamic_cast<faiss::IndexFlatIP *>(idMapIndex->index);
                    return true; // Бліц-запуск успішний! Пропускаємо SQLite!
                }
                delete idx;
            }
        }
        catch (...)
        {
            // Якщо файл пошкоджений, ігноруємо помилку та йдемо на фолбек до SQLite
        }
    }

    // 2. ФОЛБЕК (Аварійний режим): Якщо бінарника немає, будуємо індекс по-старому з SQLite
    baseIndex = new faiss::IndexFlatIP(512);
    idMapIndex = new faiss::IndexIDMap(baseIndex);
    idMapIndex->own_fields = true; // Кажемо мапі самостійно видаляти внутрішній індекс у майбутньому

    const char *sqlSelect = "SELECT id, embedding FROM scanned_files;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sqlSelect, -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    std::vector<float> allEmbeddings;
    std::vector<faiss::idx_t> allIds;

    while (sqlite3_step(stmt) == SQLITE_ROW)
    {
        int64_t id = sqlite3_column_int64(stmt, 0);
        std::string embStr = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 1));
        try
        {
            json j = json::parse(embStr);
            std::vector<float> emb = j.get<std::vector<float>>();
            if (emb.size() == 512)
            {
                allEmbeddings.insert(allEmbeddings.end(), emb.begin(), emb.end());
                allIds.push_back(id);
            }
        }
        catch (...)
        {
            continue;
        }
    }
    sqlite3_finalize(stmt);

    if (!allIds.empty())
    {
        idMapIndex->add_with_ids(allIds.size(), allEmbeddings.data(), allIds.data());
        saveIndex(); // Відразу запишемо згенерований бінарник, щоб наступний запуск був швидким
    }
    return true;
}

bool Database::saveIndex()
{
    if (!dbReady || !idMapIndex || idMapIndex->ntotal == 0)
        return false;
    try
    {
        // Миттєво скидаємо всю структуру FAISS з оперативки у файл
        faiss::write_index(idMapIndex, m_indexPath.c_str());
        return true;
    }
    catch (...)
    {
        lastError = "Не вдалося зберегти бінарний індекс FAISS на диск.";
        return false;
    }
}

bool Database::enroll(const std::string &filePath, int pageNum, const std::vector<float> &embedding, const std::string &name, const std::string &cachePath)
{
    if (!dbReady || embedding.size() != 512)
        return false;
    json j = embedding;
    std::string embStr = j.dump();

    const char *sqlInsert = "INSERT INTO scanned_files (file_path, page_num, embedding, name, cache_path) VALUES (?, ?, ?, ?, ?);";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sqlInsert, -1, &stmt, nullptr) != SQLITE_OK)
        return false;

    sqlite3_bind_text(stmt, 1, filePath.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 2, pageNum);
    sqlite3_bind_text(stmt, 3, embStr.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 4, name.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 5, cachePath.c_str(), -1, SQLITE_TRANSIENT);

    if (sqlite3_step(stmt) != SQLITE_DONE)
    {
        sqlite3_finalize(stmt);
        return false;
    }
    sqlite3_finalize(stmt);

    faiss::idx_t newId = sqlite3_last_insert_rowid(db);
    idMapIndex->add_with_ids(1, embedding.data(), &newId);
    return true;
}

bool Database::updateName(int64_t id, const std::string &newName)
{
    const char *sql = "UPDATE scanned_files SET name = ? WHERE id = ?;";
    sqlite3_stmt *stmt;

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK)
    {
        lastError = sqlite3_errmsg(db);
        return false;
    }

    sqlite3_bind_text(stmt, 1, newName.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int64(stmt, 2, id);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);

    if (!success)
    {
        lastError = sqlite3_errmsg(db);
    }

    sqlite3_finalize(stmt);
    // 💡 Примітка: Зміна імені в SQLite НЕ потребує перезбереження індексу FAISS,
    // оскільки FAISS зберігає тільки математичні вектори та числовий ID. Ім'я залишається в SQLite.
    return success;
}

std::vector<SearchResult> Database::search(const std::vector<float> &queryEmbedding, float threshold, int maxResults)
{
    std::vector<SearchResult> results;
    if (!dbReady || queryEmbedding.size() != 512 || idMapIndex->ntotal == 0)
        return results;

    int k = std::min((int)idMapIndex->ntotal, maxResults);
    std::vector<float> distances(k);
    std::vector<faiss::idx_t> labels(k);

    idMapIndex->search(1, queryEmbedding.data(), k, distances.data(), labels.data());

    for (int i = 0; i < k; ++i)
    {
        if (labels[i] != -1 && distances[i] >= threshold)
        {
            std::string fPath, name, cachePath;
            int pNum = 0;
            if (getFileInfoById(labels[i], fPath, pNum, name, cachePath))
            {
                results.push_back({labels[i], fPath, pNum, distances[i], name, cachePath});
            }
        }
    }
    return results;
}

bool Database::getFileInfoById(int64_t id, std::string &filePath, int &pageNum, std::string &name, std::string &cachePath)
{
    bool success = false;
    const char *sql = "SELECT file_path, page_num, name, cache_path FROM scanned_files WHERE id = ?;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK)
    {
        sqlite3_bind_int64(stmt, 1, id);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            filePath = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
            pageNum = sqlite3_column_int(stmt, 1);

            const char *nText = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 2));
            name = nText ? nText : "";

            const char *cText = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 3));
            cachePath = cText ? cText : "";

            success = true;
        }
    }
    sqlite3_finalize(stmt);
    return success;
}

std::vector<float> Database::getEmbeddingById(int64_t id)
{
    std::vector<float> emb;
    const char *sql = "SELECT embedding FROM scanned_files WHERE id = ?;";
    sqlite3_stmt *stmt;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK)
    {
        sqlite3_bind_int64(stmt, 1, id);
        if (sqlite3_step(stmt) == SQLITE_ROW)
        {
            std::string embStr = reinterpret_cast<const char *>(sqlite3_column_text(stmt, 0));
            try
            {
                json j = json::parse(embStr);
                emb = j.get<std::vector<float>>();
            }
            catch (...)
            {
            }
        }
    }
    sqlite3_finalize(stmt);
    return emb;
}