#define NOMINMAX

#include <QApplication>
#include <QMainWindow>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QLabel>
#include <QPushButton>
#include <QSlider>
#include <QScrollArea>
#include <QFileDialog>
#include <QMessageBox>
#include <QMenu>
#include <QAction>
#include <QDialog>
#include <QMenuBar>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QMimeData>
#include <QDesktopServices>
#include <QUrl>
#include <QMouseEvent>
#include <QPainter>
#include <QDirIterator>
#include <QTextBrowser>
#include <QProgressBar>
#include <QThread>
#include <QInputDialog>
#include <QUuid>
#include <QListWidget>
#include <QCheckBox>
#include <map>
#include <opencv2/opencv.hpp>
#include <QClipboard>
#include <QTimer>
#include <QDir>
#include <QFileInfo>
#include <QDebug>

// Легкі C-бібліотеки для парсингу документів In-Memory
#include <zip.h>
#include <fpdfview.h>
#include <fpdf_edit.h>

#include "core/FaceEngine.h"
#include "core/Database.h"

// ==========================================
// 1. СЛОВНИК МОВ ІНТЕРФЕЙСУ
// ==========================================
enum Language
{
    UKRAINIAN,
    ENGLISH
};
Language currentLang = UKRAINIAN;

std::map<std::string, std::map<Language, QString>> TR = {
    {"drop_lbl", {{UKRAINIAN, "КЛІКНІТЬ АБО ПЕРЕТЯГНІТЬ СЮДИ\nФОТО, PDF ЧИ DOCX"}, {ENGLISH, "CLICK OR DRAG & DROP HERE\nPHOTO, PDF OR DOCX"}}},
    {"slider_lbl", {{UKRAINIAN, "Поріг схожості: %1%"}, {ENGLISH, "Similarity threshold: %1%"}}},
    {"chk_group", {{UKRAINIAN, "Групувати  дублікати"}, {ENGLISH, "Group duplicates"}}},
    {"btn_search", {{UKRAINIAN, "РОЗПОЧАТИ ПОШУК"}, {ENGLISH, "START SEARCH"}}},
    {"log_title", {{UKRAINIAN, "<b>Журнал системи:</b>"}, {ENGLISH, "<b>System Log:</b>"}}},
    {"menu_file", {{UKRAINIAN, "Файл"}, {ENGLISH, "File"}}},
    {"menu_update_db", {{UKRAINIAN, "📁Оновити базу"}, {ENGLISH, "📁Update Database"}}},
    {"menu_exit", {{UKRAINIAN, "🏃Вихід"}, {ENGLISH, "🏃Exit"}}},
    {"menu_settings", {{UKRAINIAN, "Налаштування"}, {ENGLISH, "Settings"}}},
    {"menu_theme", {{UKRAINIAN, "🎨 Тема оформлення"}, {ENGLISH, "🎨 Theme"}}},
    {"menu_help", {{UKRAINIAN, "Довідка"}, {ENGLISH, "Help"}}},
    {"btn_back", {{UKRAINIAN, "◀ Назад"}, {ENGLISH, "◀ Back"}}},
    {"btn_next", {{UKRAINIAN, "Вперед ▶"}, {ENGLISH, "Next ▶"}}}};

QString t(const std::string &key)
{
    return TR[key][currentLang];
}

// --- СЛОВНИК ТЕМ ОФОРМЛЕННЯ ---
std::map<QString, QString> THEMES = {
    {"Dark (Default)", "QWidget { background-color: #1e1e1e; color: #ffffff; font-family: 'Segoe UI', sans-serif; } QPushButton { background-color: #0078d4; color: white; border-radius: 5px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #005a9e; } QPushButton:disabled { background-color: #555555; color: #888888; } QScrollArea { border: none; } QCheckBox { spacing: 8px; font-weight: bold; color: #88c0d0; } QProgressBar { border: 2px solid #555; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #4caf50; width: 10px; } QListWidget { background-color: #2a2a2a; border: 1px solid #555; border-radius: 5px; padding: 5px; }"},
    {"Light", "QWidget { background-color: #f3f3f3; color: #111111; font-family: 'Segoe UI', sans-serif; } QPushButton { background-color: #0078d4; color: white; border-radius: 5px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #005a9e; } QPushButton:disabled { background-color: #cccccc; color: #888888; } QCheckBox { spacing: 8px; font-weight: bold; color: #0078d4; } QProgressBar { border: 2px solid #ccc; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #0078d4; width: 10px; } QListWidget { background-color: #ffffff; border: 1px solid #ccc; border-radius: 5px; padding: 5px; }"},
    {"Solarized Dark", "QWidget { background-color: #002b36; color: #839496; font-family: 'Segoe UI', sans-serif; } QPushButton { background-color: #2aa198; color: #002b36; border-radius: 5px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #268bd2; } QCheckBox { spacing: 8px; font-weight: bold; color: #2aa198; } QListWidget { background-color: #073642; border: 1px solid #586e75; border-radius: 5px; }"},
    {"Nord", "QWidget { background-color: #2e3440; color: #d8dee9; font-family: 'Segoe UI', sans-serif; } QPushButton { background-color: #88c0d0; color: #2e3440; border-radius: 5px; padding: 8px; font-weight: bold; } QPushButton:hover { background-color: #81a1c1; } QPushButton:disabled { background-color: #4c566a; color: #d8dee9; } QCheckBox { spacing: 8px; font-weight: bold; color: #a3be8c; } QListWidget { background-color: #3b4252; border: 1px solid #4c566a; border-radius: 5px; }"}};

cv::Mat loadMatUtf8(const QString &path)
{
    QImage qimg;
    if (!qimg.load(path))
        return cv::Mat();
    qimg = qimg.convertToFormat(QImage::Format_RGB888);
    cv::Mat mat(qimg.height(), qimg.width(), CV_8UC3, (void *)qimg.bits(), qimg.bytesPerLine());
    cv::Mat result;
    cv::cvtColor(mat, result, cv::COLOR_RGB2BGR);
    return result;
}

QPixmap cvMatToQPixmap(const cv::Mat &inMat)
{
    if (inMat.empty())
        return QPixmap();
    cv::Mat tmp;
    cv::cvtColor(inMat, tmp, cv::COLOR_BGR2RGB);
    return QPixmap::fromImage(QImage((const uchar *)tmp.data, tmp.cols, tmp.rows, tmp.step, QImage::Format_RGB888).copy());
}

// ==========================================
// КЛАСИ ДІАЛОГОВИХ ВІКОН
// ==========================================

class MultiFolderDialog : public QDialog
{
public:
    QListWidget *listWidget;
    MultiFolderDialog(QWidget *parent = nullptr) : QDialog(parent)
    {
        setWindowTitle("Вибір папок для сканування");
        resize(500, 350);
        QVBoxLayout *mainLayout = new QVBoxLayout(this);
        mainLayout->addWidget(new QLabel("<b>Список папок для індексації:</b>"));
        listWidget = new QListWidget();
        mainLayout->addWidget(listWidget);

        QHBoxLayout *btnLayout = new QHBoxLayout();
        QPushButton *btnAdd = new QPushButton("➕ Додати папку");
        connect(btnAdd, &QPushButton::clicked, [this]()
                {
            QString dir = QFileDialog::getExistingDirectory(this, "Оберіть папку");
            if (!dir.isEmpty()) {
                bool exists = false;
                for(int i = 0; i < listWidget->count(); ++i) {
                    if(listWidget->item(i)->text() == dir) { exists = true; break; }
                }
                if(!exists) listWidget->addItem(dir);
            } });
        QPushButton *btnRemove = new QPushButton("❌ Видалити вибрану");
        connect(btnRemove, &QPushButton::clicked, [this]()
                { qDeleteAll(listWidget->selectedItems()); });

        btnLayout->addWidget(btnAdd);
        btnLayout->addWidget(btnRemove);
        mainLayout->addLayout(btnLayout);

        QHBoxLayout *actionLayout = new QHBoxLayout();
        QPushButton *btnStart = new QPushButton("🚀 Почати сканування");
        btnStart->setStyleSheet("background-color: #4caf50; color: white;");
        connect(btnStart, &QPushButton::clicked, this, &QDialog::accept);

        QPushButton *btnCancel = new QPushButton("Скасувати");
        connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);

        actionLayout->addStretch();
        actionLayout->addWidget(btnCancel);
        actionLayout->addWidget(btnStart);

        mainLayout->addWidget(new QLabel(""));
        mainLayout->addLayout(actionLayout);
    }
    QStringList getSelectedFolders() const
    {
        QStringList folders;
        for (int i = 0; i < listWidget->count(); ++i)
            folders.append(listWidget->item(i)->text());
        return folders;
    }
};

class FaceSelectionDialog : public QDialog
{
public:
    int selectedIndex = -1;
    std::vector<cv::Rect2f> locs;
    float scaleFactor = 1.0f;
    QLabel *lbl;

    FaceSelectionDialog(const QString &imagePath, const std::vector<cv::Rect2f> &locations, QWidget *parent = nullptr) : QDialog(parent), locs(locations)
    {
        setWindowTitle("Знайдено кілька облич");
        QVBoxLayout *lay = new QVBoxLayout(this);
        lbl = new QLabel();
        QImage img(imagePath);
        QPixmap pix = QPixmap::fromImage(img);
        int maxSize = 800;
        if (pix.width() > maxSize || pix.height() > maxSize)
        {
            QPixmap scaledPix = pix.scaled(maxSize, maxSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            scaleFactor = (float)pix.width() / scaledPix.width();
            pix = scaledPix;
        }
        QPainter p(&pix);
        p.setPen(QPen(QColor(0, 255, 0), 3));
        for (const auto &rect : locs)
            p.drawRect(rect.x / scaleFactor, rect.y / scaleFactor, rect.width / scaleFactor, rect.height / scaleFactor);
        p.end();
        lbl->setPixmap(pix);
        lay->addWidget(lbl);
    }
    void mousePressEvent(QMouseEvent *event) override
    {
        QPoint pos = lbl->mapFrom(this, event->pos());
        for (size_t i = 0; i < locs.size(); ++i)
        {
            QRectF r(locs[i].x / scaleFactor, locs[i].y / scaleFactor, locs[i].width / scaleFactor, locs[i].height / scaleFactor);
            if (r.contains(pos))
            {
                selectedIndex = i;
                accept();
                return;
            }
        }
    }
};

class PdfFaceSelectionDialog : public QDialog
{
public:
    int selectedIndex = -1;
    PdfFaceSelectionDialog(const std::vector<cv::Mat> &crops, QWidget *parent = nullptr) : QDialog(parent)
    {
        setWindowTitle("У документі знайдено кілька обличь");
        resize(650, 450);
        QVBoxLayout *mainLayout = new QVBoxLayout(this);
        mainLayout->addWidget(new QLabel("<h3 style='color:#0078d4; margin-bottom: 10px;'>Документ містить кілька облич. Оберіть потрібне:</h3>"));

        QScrollArea *scrollArea = new QScrollArea();
        scrollArea->setStyleSheet("QScrollArea { border: 1px solid #444; border-radius: 5px; background-color: #1e1e1e; }");
        scrollArea->setWidgetResizable(true);
        QWidget *gridWidget = new QWidget();
        QGridLayout *grid = new QGridLayout(gridWidget);
        grid->setSpacing(15);

        int row = 0, col = 0;
        for (size_t i = 0; i < crops.size(); ++i)
        {
            QPushButton *btn = new QPushButton();
            btn->setFixedSize(130, 130);
            btn->setStyleSheet("QPushButton { border: 2px solid transparent; border-radius: 8px; background-color: #2a2a2a; } QPushButton:hover { border: 2px solid #4caf50; background-color: #333; }");
            QPixmap pix = cvMatToQPixmap(crops[i]).scaled(110, 110, Qt::KeepAspectRatio, Qt::SmoothTransformation);
            btn->setIcon(QIcon(pix));
            btn->setIconSize(QSize(110, 110));
            btn->setCursor(Qt::PointingHandCursor);
            btn->setToolTip(QString("Обличчя #%1").arg(i + 1));

            connect(btn, &QPushButton::clicked, [this, i]()
                    { selectedIndex = i; accept(); });
            grid->addWidget(btn, row, col);
            if (++col >= 4)
            {
                col = 0;
                row++;
            }
        }
        grid->setRowStretch(row + 1, 1);
        grid->setColumnStretch(4, 1);
        scrollArea->setWidget(gridWidget);
        mainLayout->addWidget(scrollArea);

        QPushButton *btnCancel = new QPushButton("Скасувати");
        btnCancel->setStyleSheet("background-color: #555555; color: white; padding: 8px; border-radius: 5px; font-weight: bold;");
        connect(btnCancel, &QPushButton::clicked, this, &QDialog::reject);

        QHBoxLayout *btnLayout = new QHBoxLayout();
        btnLayout->addStretch();
        btnLayout->addWidget(btnCancel);
        mainLayout->addLayout(btnLayout);
    }
};

class ComparisonDialog : public QDialog
{
public:
    ComparisonDialog(const QString &targetPath, const cv::Mat &targetFace, const SearchResult &resultData, std::shared_ptr<Database> db, QWidget *parent = nullptr) : QDialog(parent)
    {
        setWindowTitle("Порівняння облич");
        resize(940, 580);

        QVBoxLayout *mainLayout = new QVBoxLayout(this);
        mainLayout->setContentsMargins(20, 20, 20, 20);
        mainLayout->setSpacing(15);

        QHBoxLayout *contentLayout = new QHBoxLayout();
        contentLayout->setSpacing(20);

        QVBoxLayout *leftCol = new QVBoxLayout();
        QLabel *lblTitleLeft = new QLabel("<h3 align='center' style='color: #0078d4; margin-bottom: 5px; font-weight: bold;'>🔵 ФОТО-ЗАПИТ</h3>");
        QLabel *lblTarget = new QLabel();
        if (!targetFace.empty())
            lblTarget->setPixmap(cvMatToQPixmap(targetFace).scaled(370, 370, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        lblTarget->setAlignment(Qt::AlignCenter);
        lblTarget->setStyleSheet("border: 3px solid #0078d4; border-radius: 8px; background-color: #121212; padding: 4px;");
        lblTarget->setFixedSize(380, 380);

        lblTarget->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(lblTarget, &QLabel::customContextMenuRequested, [this, targetPath](const QPoint &pos)
                {
            Q_UNUSED(pos); QMenu menu; QAction* actOpen = menu.addAction("📁 Відкрити папку з файлом");
            if (menu.exec(QCursor::pos()) == actOpen) QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(targetPath).absolutePath())); });

        leftCol->addWidget(lblTitleLeft);
        leftCol->addWidget(lblTarget);

        QLabel *lblSim = new QLabel(QString("<div style='background-color: #2a2a2a; border: 1px solid #444444; border-radius: 10px; padding: 15px; text-align: center;'><b style='font-size: 34px; color: #4caf50;'>%1%</b><br><span style='color: #aaaaaa; font-size: 11px; font-weight: bold; letter-spacing: 1px;'>СХОЖІСТЬ</span></div>").arg(resultData.similarity * 100.0f, 0, 'f', 1));
        lblSim->setAlignment(Qt::AlignCenter);

        QVBoxLayout *rightCol = new QVBoxLayout();
        QString displayName = resultData.name.empty() ? "Особа без імені" : QString::fromStdString(resultData.name);
        QLabel *lblTitleRight = new QLabel(QString("<h3 align='center' style='color: #4caf50; margin-bottom: 5px; font-weight: bold;'>🟢 %1</h3>").arg(displayName));

        QLabel *lblRes = new QLabel();
        QImage resImg;
        QString relOrAbsPath = QString::fromStdString(resultData.cachePath);
        QString absCache = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(relOrAbsPath);

        if (!resultData.cachePath.empty() && QFile::exists(absCache))
            resImg.load(absCache);
        else if (!resultData.cachePath.empty() && QFile::exists(relOrAbsPath))
            resImg.load(relOrAbsPath);
        else
            resImg.load(QString::fromStdString(resultData.filePath));

        if (!resImg.isNull())
            lblRes->setPixmap(QPixmap::fromImage(resImg).scaled(370, 370, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        else
            lblRes->setText("ФОТО\nНЕ ЗНАЙДЕНО");

        lblRes->setAlignment(Qt::AlignCenter);
        lblRes->setStyleSheet("border: 3px solid #4caf50; border-radius: 8px; background-color: #121212; padding: 4px;");
        lblRes->setFixedSize(380, 380);

        lblRes->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(lblRes, &QLabel::customContextMenuRequested, [this, resultData, db, lblTitleRight](const QPoint &pos)
                {
            Q_UNUSED(pos); QMenu menu;
            QAction* actName = menu.addAction("✏️ Додати ім'я в БД");
            QAction* actOpen = menu.addAction("📁 Відкрити папку з файлом");
            QAction* sel = menu.exec(QCursor::pos());
            if (sel == actOpen) {
                QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(QString::fromStdString(resultData.filePath)).absolutePath()));
            } else if (sel == actName) {
                bool ok; QString text = QInputDialog::getText(this, "Додати ім'я", "Введіть ім'я особи в БД:", QLineEdit::Normal, QString::fromStdString(resultData.name), &ok);
                if (ok) {
                    db->updateName(resultData.id, text.toStdString());
                    QString newTitle = text.isEmpty() ? QFileInfo(QString::fromStdString(resultData.filePath)).fileName() : text;
                    lblTitleRight->setText(QString("<h3 align='center' style='color: #4caf50; margin-bottom: 5px; font-weight: bold;'>🟢 %1</h3>").arg(newTitle));
                }
            } });

        rightCol->addWidget(lblTitleRight);
        rightCol->addWidget(lblRes);

        contentLayout->addLayout(leftCol);
        contentLayout->addWidget(lblSim);
        contentLayout->addLayout(rightCol);
        mainLayout->addLayout(contentLayout);

        QPushButton *btnCopy = new QPushButton("📋 Скопіювати порівняння");
        btnCopy->setFixedHeight(42);
        btnCopy->setStyleSheet("background-color: #0078d4; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");

        connect(btnCopy, &QPushButton::clicked, [this, btnCopy]()
                {
            btnCopy->hide(); QApplication::processEvents();
            QPixmap screenshot = this->grab();
            btnCopy->show(); QApplication::clipboard()->setPixmap(screenshot);
            btnCopy->setText("✅ Скопійовано!");
            btnCopy->setStyleSheet("background-color: #4caf50; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");
            QTimer::singleShot(1500, [btnCopy]() {
                btnCopy->setText("📋 Скопіювати порівняння");
                btnCopy->setStyleSheet("background-color: #0078d4; color: white; font-weight: bold; font-size: 13px; border-radius: 5px;");
            }); });

        mainLayout->addWidget(btnCopy);
    }
};

// ==========================================
// ФОНОВИЙ ПОТІК (СКАНЕР)
// ==========================================
class DBWorker : public QObject
{
    Q_OBJECT
public:
    DBWorker(const QStringList &dirPaths, std::shared_ptr<FaceEngine> engine, std::shared_ptr<Database> db, const QString &appDirPath)
        : m_dirPaths(dirPaths), m_engine(engine), m_db(db), m_appDirPath(appDirPath) {}

public slots:
    void process()
    {
        emit logMsg(QString("<font color='#88c0d0'>[WORKER]</font> Пошук запущено. Аналіз папок: <b>%1 шт.</b>").arg(m_dirPaths.size()));
        QString cacheDir = m_appDirPath + "/cache_faces";
        QDir().mkpath(cacheDir);

        QStringList files;
        for (const QString &dirPath : m_dirPaths)
        {
            QDirIterator it(dirPath, QDir::Files, QDirIterator::Subdirectories);
            while (it.hasNext())
            {
                QString file = it.next();
                QString ext = QFileInfo(file).suffix().toLower();
                if (ext == "jpg" || ext == "jpeg" || ext == "png" || ext == "pdf" ||
                    ext == "webp" || ext == "tiff" || ext == "tif" || ext == "bmp" || ext == "docx")
                {
                    files.append(file);
                }
            }
        }

        int totalFiles = files.size();
        if (totalFiles == 0)
        {
            emit finishedScan(0, 0);
            return;
        }

        int count = 0, normalAdded = 0, enhancedAdded = 0;
        const size_t BATCH_SIZE = 8;

        struct FaceTask
        {
            QString origPath;
            int pageNum;
            cv::Mat cropImg;
            QString cachePath;
            bool isEnhanced;
        };
        std::vector<FaceTask> faceBatch;

        auto flushBatch = [&]()
        {
            if (faceBatch.empty())
                return;
            std::vector<cv::Mat> crops;
            for (const auto &task : faceBatch)
                crops.push_back(task.cropImg);

            std::vector<std::vector<float>> embeddings;
            if (m_engine->extractEmbeddingsBatch(crops, embeddings))
            {
                for (size_t i = 0; i < faceBatch.size(); ++i)
                {
                    m_db->enroll(faceBatch[i].origPath.toStdString(), faceBatch[i].pageNum, embeddings[i], "", faceBatch[i].cachePath.toStdString());
                    if (faceBatch[i].isEnhanced)
                        enhancedAdded++;
                    else
                        normalAdded++;
                }
            }
            else
            {
                emit logMsg(QString("<font color='red'>[ПОМИЛКА]</font> Збій Batch Inference! Причина: %1").arg(QString::fromStdString(m_engine->getLastError())));
            }
            faceBatch.clear();
        };

        auto processSingleImage = [&](const cv::Mat &origImg, const QString &sourcePath, int pageNum)
        {
            auto faces = m_engine->detectFaces(origImg, 0.5f);
            bool usedEnhancement = false;

            if (faces.empty())
            {
                cv::Mat lab, enhanced, blurred;
                cv::cvtColor(origImg, lab, cv::COLOR_BGR2Lab);
                std::vector<cv::Mat> lab_planes(3);
                cv::split(lab, lab_planes);
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
                clahe->apply(lab_planes[0], lab_planes[0]);
                cv::merge(lab_planes, lab);
                cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
                cv::GaussianBlur(enhanced, blurred, cv::Size(0, 0), 3);
                cv::addWeighted(enhanced, 1.5, blurred, -0.5, 0, enhanced);

                faces = m_engine->detectFaces(enhanced, 0.45f);
                if (!faces.empty())
                    usedEnhancement = true;
            }

            if (faces.empty())
            {
                emit logMsg(QString("<font color='#888888'>[ІНФО]</font> Пропущено %1 (облич не виявлено).").arg(QFileInfo(sourcePath).fileName()));
                return;
            }

            for (auto &face : faces)
            {
                cv::Mat alignedCrop = m_engine->alignFace(origImg, face);
                if (!alignedCrop.empty())
                {
                    QString fileName = QUuid::createUuid().toString(QUuid::WithoutBraces) + ".jpg";
                    QString relCachePath = "cache_faces/" + fileName;
                    QString absCachePath = QDir(cacheDir).absoluteFilePath(fileName);
                    cvMatToQPixmap(alignedCrop).toImage().save(absCachePath, "JPG");
                    faceBatch.push_back({sourcePath, pageNum, alignedCrop, relCachePath, usedEnhancement});

                    if (faceBatch.size() >= BATCH_SIZE)
                        flushBatch();
                }
            }
        };

        for (const QString &path : files)
        {
            if (m_db->isFileIndexed(path.toStdString()))
            {
                count++;
                emit progressUpdate(count, totalFiles);
                continue;
            }
            QString ext = QFileInfo(path).suffix().toLower();

            if (ext == "pdf")
            {
                FPDF_DOCUMENT doc = FPDF_LoadDocument(path.toUtf8().constData(), nullptr);
                if (!doc)
                {
                    emit logMsg(QString("<font color='orange'>[ІНФО]</font> Пошкоджений PDF: %1. Пропущено.").arg(QFileInfo(path).fileName()));
                    count++;
                    emit progressUpdate(count, totalFiles);
                    continue;
                }

                int pageCount = FPDF_GetPageCount(doc);
                for (int i = 0; i < pageCount; ++i)
                {
                    FPDF_PAGE page = FPDF_LoadPage(doc, i);
                    if (!page)
                        continue;

                    double widthPoints = FPDF_GetPageWidth(page);
                    double heightPoints = FPDF_GetPageHeight(page);
                    float scale = 150.0f / 72.0f;
                    int width = static_cast<int>(widthPoints * scale);
                    int height = static_cast<int>(heightPoints * scale);

                    cv::Mat pageBGRA(height, width, CV_8UC4, cv::Scalar(255, 255, 255, 255));
                    FPDF_BITMAP bitmap = FPDFBitmap_CreateEx(width, height, FPDFBitmap_BGRA, pageBGRA.data, static_cast<int>(pageBGRA.step));

                    if (bitmap)
                    {
                        FPDF_RenderPageBitmap(bitmap, page, 0, 0, width, height, 0, 0);
                        cv::Mat bgrImg;
                        cv::cvtColor(pageBGRA, bgrImg, cv::COLOR_BGRA2BGR);
                        if (!bgrImg.empty())
                            processSingleImage(bgrImg, path, i + 1);
                        FPDFBitmap_Destroy(bitmap);
                    }
                    FPDF_ClosePage(page);
                }
                FPDF_CloseDocument(doc);
            }
            else if (ext == "docx")
            {
                int err = 0;
                zip_t *archive = zip_open(path.toUtf8().constData(), 0, &err);
                if (archive)
                {
                    zip_int64_t num_entries = zip_get_num_entries(archive, 0);
                    for (zip_int64_t i = 0; i < num_entries; i++)
                    {
                        const char *name = zip_get_name(archive, i, 0);
                        if (!name)
                            continue;
                        QString fileName(name);
                        if (fileName.startsWith("word/media/", Qt::CaseInsensitive))
                        {
                            zip_stat_t sb;
                            if (zip_stat_index(archive, i, 0, &sb) == 0)
                            {
                                zip_file_t *f = zip_fopen_index(archive, i, 0);
                                if (f)
                                {
                                    QByteArray data;
                                    data.resize(sb.size);
                                    zip_fread(f, data.data(), sb.size);
                                    zip_fclose(f);
                                    QImage qimg = QImage::fromData(data);
                                    if (!qimg.isNull())
                                    {
                                        qimg = qimg.convertToFormat(QImage::Format_RGB888);
                                        cv::Mat img(qimg.height(), qimg.width(), CV_8UC3, (void *)qimg.bits(), qimg.bytesPerLine());
                                        cv::Mat bgrImg;
                                        cv::cvtColor(img, bgrImg, cv::COLOR_RGB2BGR);
                                        if (!bgrImg.empty())
                                            processSingleImage(bgrImg, path, 0);
                                    }
                                }
                            }
                        }
                    }
                    zip_close(archive);
                }
                else
                {
                    emit logMsg(QString("<font color='orange'>[ІНФО]</font> Не вдалося прочитати DOCX: %1").arg(QFileInfo(path).fileName()));
                }
            }
            else
            {
                cv::Mat img = loadMatUtf8(path);
                if (!img.empty())
                    processSingleImage(img, path, 0);
                else
                    emit logMsg(QString("<font color='orange'>[ІНФО]</font> Qt не зміг прочитати: %1 (Відсутній кодек)").arg(QFileInfo(path).fileName()));
            }
            count++;
            emit progressUpdate(count, totalFiles);
            if (count % 10 == 0)
                emit logMsg(QString("[ПРОГРЕС] Опрацьовано %1 з %2 файлів...").arg(count).arg(totalFiles));
        }

        if (!faceBatch.empty())
            flushBatch();
        emit logMsg(QString("<font color='green'>[СИСТЕМА]</font> Фонове сканування завершено! Всього додано облич: <b>%1</b>").arg(normalAdded + enhancedAdded));
        emit finishedScan(normalAdded, enhancedAdded);
    }
signals:
    void logMsg(QString msg);
    void progressUpdate(int current, int total);
    void finishedScan(int normalCount, int enhancedCount);

private:
    QStringList m_dirPaths;
    std::shared_ptr<FaceEngine> m_engine;
    std::shared_ptr<Database> m_db;
    QString m_appDirPath;
};

// ==========================================
// ГОЛОВНЕ ВІКНО ДОДАТКУ
// ==========================================
class FaceFinderApp : public QMainWindow
{
public:
    FaceFinderApp()
    {
        QString basePath = QApplication::applicationDirPath() + "/models/";
        engine = std::make_shared<FaceEngine>(QDir(basePath).absoluteFilePath("det_10g.onnx").toStdWString(), QDir(basePath).absoluteFilePath("w600k_r50.onnx").toStdWString(), false);
        db = std::make_shared<Database>((QApplication::applicationDirPath() + "/faces.sqlite").toStdString());

        initUI();
        setStyleSheet(THEMES["Dark (Default)"]);
        resize(1150, 750);
        retranslateUI(); // Ініціалізуємо переклади
    }

private:
    std::shared_ptr<FaceEngine> engine;
    std::shared_ptr<Database> db;

    QLabel *dropLabel;
    QSlider *sld;
    QLabel *sldLabel;
    QCheckBox *chkGroup;
    QPushButton *btnSearch;
    QTextBrowser *logConsole;
    QProgressBar *scanProgress;
    QGridLayout *resGrid;
    QLabel *logTitleLbl;

    std::vector<SearchResult> m_currentResults;
    int m_currentPage = 0;
    const int RESULTS_PER_PAGE = 16;
    QPushButton *btnPrevPage;
    QPushButton *btnNextPage;
    QLabel *lblPageInfo;

    std::vector<float> selectedEnc;
    cv::Rect2f selectedLoc;
    cv::Mat m_targetFaceCrop;
    QString targetPath;

    QThread *workerThread = nullptr;
    DBWorker *worker = nullptr;

    QMenu *fileMenu;
    QAction *actUpdateDb;
    QAction *actExit;
    QMenu *settingsMenu;
    QMenu *themeSub;
    QMenu *helpMenu;

    void initUI()
    {
        QMenuBar *bar = menuBar();

        fileMenu = bar->addMenu("");
        actUpdateDb = fileMenu->addAction("", this, [this]()
                                          { updateDB(); });
        fileMenu->addSeparator();
        actExit = fileMenu->addAction("", this, &QWidget::close);

        settingsMenu = bar->addMenu("");
        QMenu *langSub = settingsMenu->addMenu("🌐 Мова / Language");
        langSub->addAction("🇺🇦 Українська", this, [this]()
                           { currentLang = UKRAINIAN; retranslateUI(); });
        langSub->addAction("🇺🇸 English", this, [this]()
                           { currentLang = ENGLISH; retranslateUI(); });

        themeSub = settingsMenu->addMenu("");
        for (const auto &pair : THEMES)
        {
            themeSub->addAction(pair.first, this, [this, name = pair.first]()
                                { setStyleSheet(THEMES[name]); });
        }

        QAction *gpuAction = settingsMenu->addAction("🚀 GPU (CUDA / TensorRT)");
        gpuAction->setCheckable(true);
        gpuAction->setChecked(false);
        connect(gpuAction, &QAction::triggered, this, [this, gpuAction](bool checked)
                {
            logConsole->append(checked ? "<font color='#e3a828'>[СИСТЕМА]</font> Спроба перемикання на GPU (CUDA)..." : "<font color='#e3a828'>[СИСТЕМА]</font> Перемикання на процесор (CPU)...");
            QApplication::processEvents(); 
            QString basePath = QApplication::applicationDirPath() + "/models/";
            engine = std::make_shared<FaceEngine>(QDir(basePath).absoluteFilePath("det_10g.onnx").toStdWString(), QDir(basePath).absoluteFilePath("w600k_r50.onnx").toStdWString(), checked);
            if (engine->isLoaded()) {
                logConsole->append(checked ? "<font color='green'>[СИСТЕМА]</font> <b>Апаратне прискорення GPU активовано успішно!</b>" : "<font color='green'>[СИСТЕМА]</font> <b>Режим CPU активовано.</b>");
            } else {
                logConsole->append("<font color='red'>[ПОМИЛКА]</font> Не вдалося запустити GPU: " + QString::fromStdString(engine->getLastError()));
                if (checked) {
                    gpuAction->setChecked(false);
                    engine = std::make_shared<FaceEngine>(QDir(basePath).absoluteFilePath("det_10g.onnx").toStdWString(), QDir(basePath).absoluteFilePath("w600k_r50.onnx").toStdWString(), false);
                }
            } });
        settingsMenu->addSeparator();
        settingsMenu->addAction("⚠️ Очистити базу повністю", this, [this]()
                                {
            auto reply = QMessageBox::question(this, "Очищення", "Ви впевнені, що хочете повністю стерти базу данних та фото-мініатюри?", QMessageBox::Yes | QMessageBox::No);
            if (reply == QMessageBox::Yes) {
                QString cacheDirPath = QApplication::applicationDirPath() + "/cache_faces";
                QDir cacheDir(cacheDirPath);
                if (cacheDir.exists()) {
                    QStringList files = cacheDir.entryList(QDir::Files);
                    for (const QString& filename : files) cacheDir.remove(filename);
                }
                db->clearDatabase();
                logConsole->append("<font color='red'>[БД]</font> Базу даних повністю очищено.");
                QMessageBox::information(this, "Успіх", "Базу даних та фото-мініатюри успішно видалено!");
            } });

        helpMenu = bar->addMenu("");
        helpMenu->addAction("📖 Інструкція користувача", this, [this]()
                            {
            QDialog dlg(this); dlg.setWindowTitle("Інструкція користувача FaceFin"); dlg.resize(750, 550);
            QVBoxLayout layout(&dlg); QTextBrowser text;
            text.setHtml(QString::fromUtf8(R"(
                <h2 style='color:#0078d4;'>Інструкція користувача FaceFin</h2>
                <h3 style='color:#4caf50;'>1. Первинне налаштування та індексація</h3>
                <p>Перед початком пошуку програмі необхідно проаналізувати файли. Натисніть <b>Файл -> Оновити базу</b>. У вікні додайте папки, де зберігаються фотографії або документи, та натисніть "Почати сканування". Програма автоматично знайде всі обличчя і збереже їх математичні вектори в базу даних.</p>
                <h3 style='color:#4caf50;'>2. Як здійснювати пошук?</h3>
                <p>Просто <b>перетягніть</b> фотографію, PDF-документ або файл MS Word (.docx) у пунктирне вікно зліва (або клікніть по ньому, щоб вибрати файл через провідник).</p>
                <ul>
                    <li>Якщо на фото одне обличчя — воно одразу додасться для пошуку.</li>
                    <li>Якщо облич кілька (групове фото чи документ) — програма покаже вікно, де ви зможете клікнути на потрібну людину.</li>
                </ul>
                <p>Після цього тисніть кнопку <b>РОЗПОЧАТИ ПОШУК</b>.</p>
                <h3 style='color:#4caf50;'>3. Робота з результатами</h3>
                <ul>
                    <li><b>Поріг схожості:</b> Чим вищий відсоток, тим прискіпливіший пошук. 45-50% — ідеально для пошуку тієї ж людини в різному віці. 70%+ — для пошуку ідентичних фото.</li>
                    <li><b>Групувати дублікати:</b> Якщо галочка активна, програма сховає ідентичні фотографії в одну картку.</li>
                </ul>
                <h3 style='color:#4caf50;'>4. Меню знайденого обличчя (Правий клік)</h3>
                <p>Натисніть правою кнопкою миші на будь-яку картку в результатах:</p>
                <ul>
                    <li><b>✏️ Додати ім'я:</b> Дозволяє підписати людину в базі.</li>
                    <li><b>🔍 Порівняти фото:</b> Відкриває велике вікно детального порівняння вашого запиту із знайденим файлом.</li>
                    <li><b>📁 Відкрити папку:</b> Відкриє папку Windows, виділивши файл-оригінал.</li>
                </ul>
            )"));
            layout.addWidget(&text); dlg.exec(); });

        helpMenu->addAction("✨ Що нового", this, [this]()
                            {
            QDialog dlg(this); dlg.setWindowTitle("Історія оновлень FaceFin"); dlg.resize(650, 500);
            QVBoxLayout layout(&dlg); QTextBrowser text;
            text.setHtml(QString::fromUtf8(R"(
                <h2 style='color:#0078d4;'>Що нового у FaceFin (v0.9.8b)</h2>
                <h3 style='color:#4caf50;'>Поточне оновлення (v0.9.8b)</h3>
                <ul>
                    <li><b>Групування дублікатів:</b> Інтелектуальна система, яка обєднує 100% однакові фотографії з різних папок в одну картку.</li>
                    <li><b>Підтримка MS Word:</b>  швидкий парсинг .docx документів через бібліотеку libzip.</li>
                    <li><b>Блискавичний запуск:</b> Завдяки бінарній серіалізації FAISS, програма запускається за мілісекунди.</li>
                    <li><b>Google PDFium:</b> рендеринг PDF документів.</li>
                </ul>
                <hr>
                <h3>Попередні етапи розробки (v0.6 - v0.9)</h3>
                <ul>
                    <li><b>v0.9.0:</b> Повністю перероблено архітектуру збереження облич. Перехід на унікальні ID.</li>
                    <li><b>v0.8.5:</b> Покращене вікно візуального порівняння.</li>
                    <li><b>v0.8.0:</b> Впроваджено систему <b>CLAHE</b> для автоматичного висвітлення глибоких тіней.</li>
                    <li><b>v0.7.5:</b> Апаратне прискорення GPU (CUDA/TensorRT).</li>
                    <li><b>v0.6.0:</b> Базова реалізація: виявлення облич, генерація ШІ-векторів та побудова БД.</li>
                </ul>
            )"));
            layout.addWidget(&text); dlg.exec(); });

        helpMenu->addAction("ℹ️ Про програму", this, [this]()
                            {
            QDialog dlg(this); dlg.setWindowTitle("Про FaceFin"); dlg.resize(600, 450);
            QVBoxLayout layout(&dlg); QTextBrowser text;
            text.setHtml(QString::fromUtf8(R"(
                <h1 style='color:#0078d4; margin-bottom: 0px;'>FaceFin</h1>
                <p style='margin-top: 0px;'>Версія: <b>0.9.8b</b></p>
                <p><b>FaceFin</b> — це система інтелектуального пошуку та біометричного порівняння облич. Працює виключно локально.</p>
                <h3 style='color:#4caf50;'>Нейромережі</h3>
                <ul>
                    <li><b>RetinaFace (det_10g.onnx):</b> Точне знаходження облич.</li>
                    <li><b>ArcFace (w600k_r50.onnx):</b> Генерація унікального біометричного вектора.</li>
                </ul>
                <h3 style='color:#4caf50;'>Використані технології</h3>
                <ul>
                    <li><b>C++ / Qt5:</b> Ядро та GUI.</li>
                    <li><b>OpenCV 4.x:</b> Обробка зображень.</li>
                    <li><b>ONNX Runtime:</b> Движок нейромереж.</li>
                    <li><b>Meta FAISS:</b> Векторний пошук.</li>
                    <li><b>SQLite 3:</b> Збереження шляхів.</li>
                    <li><b>Google PDFium / libzip:</b> Парсинг документів docx та pdf.</li>
                </ul>
                <hr>
                <p><b>АВТОР ТА РОЗРОБНИК:</b> AlinaKas94</p>
                <p style='font-size: 11px; color: #888888;'>Програмне забезпечення базується на компонентах з відкритим вихідним кодом. Усі права на використані бібліотеки (OpenCV, Qt, FAISS, ONNX) належать їх відповідним авторам згідно з ліцензіями Apache 2.0, MIT та (L)GPL.</p>
            )"));
            layout.addWidget(&text); dlg.exec(); });

        QWidget *mainWidget = new QWidget();
        setCentralWidget(mainWidget);
        QHBoxLayout *layout = new QHBoxLayout(mainWidget);

        QVBoxLayout *leftPanel = new QVBoxLayout();
        leftPanel->setSpacing(15);
        leftPanel->setContentsMargins(0, 0, 10, 0);

        dropLabel = new QLabel();
        dropLabel->setAlignment(Qt::AlignCenter);
        dropLabel->setStyleSheet("border: 2px dashed #777; padding: 40px; font-weight: bold; font-size: 14px;");
        dropLabel->setCursor(Qt::PointingHandCursor);
        dropLabel->setAcceptDrops(true);
        dropLabel->installEventFilter(this);
        leftPanel->addWidget(dropLabel);

        sld = new QSlider(Qt::Horizontal);
        sld->setRange(20, 95);
        sld->setValue(45);
        sldLabel = new QLabel();
        connect(sld, &QSlider::valueChanged, [this](int v)
                { sldLabel->setText(t("slider_lbl").arg(v)); });
        leftPanel->addWidget(sldLabel);
        leftPanel->addWidget(sld);

        chkGroup = new QCheckBox();
        chkGroup->setChecked(true);
        leftPanel->addWidget(chkGroup);

        btnSearch = new QPushButton();
        btnSearch->setEnabled(false);
        btnSearch->setFixedHeight(50);
        connect(btnSearch, &QPushButton::clicked, this, &FaceFinderApp::runSearch);
        leftPanel->addWidget(btnSearch);

        scanProgress = new QProgressBar();
        scanProgress->setValue(0);
        scanProgress->hide();
        leftPanel->addWidget(scanProgress);

        logConsole = new QTextBrowser();
        logConsole->setStyleSheet("font-family: monospace; padding: 5px;");
        logConsole->append("<font color='#4caf50'>[СИСТЕМА]</font> Програму успішно запущено.");
        logConsole->append(db->isReady() ? "<font color='green'>[БД]</font> База даних підключена успішно." : "<font color='red'>[БД]</font> Помилка підключення бази!");
        logConsole->append(QString("<font color='#88c0d0'>[БД]</font> Всього проіндексовано облич у базі: <b>%1</b>").arg(db->getRecordCount()));

        logTitleLbl = new QLabel();
        leftPanel->addWidget(logTitleLbl);
        leftPanel->addWidget(logConsole);

        layout->addLayout(leftPanel, 1);

        QVBoxLayout *rightPanel = new QVBoxLayout();
        QScrollArea *scroll = new QScrollArea();
        scroll->setStyleSheet("border: none;");
        QWidget *scrollContent = new QWidget();
        resGrid = new QGridLayout(scrollContent);
        resGrid->setAlignment(Qt::AlignTop | Qt::AlignLeft);
        scroll->setWidgetResizable(true);
        scroll->setWidget(scrollContent);
        rightPanel->addWidget(scroll);

        QHBoxLayout *pageLayout = new QHBoxLayout();
        btnPrevPage = new QPushButton();
        btnNextPage = new QPushButton();
        lblPageInfo = new QLabel();
        lblPageInfo->setAlignment(Qt::AlignCenter);

        btnPrevPage->setEnabled(false);
        btnNextPage->setEnabled(false);

        connect(btnPrevPage, &QPushButton::clicked, [this]()
                { if(m_currentPage > 0) { m_currentPage--; displayPage(); } });
        connect(btnNextPage, &QPushButton::clicked, [this]()
                { if((m_currentPage + 1) * RESULTS_PER_PAGE < m_currentResults.size()) { m_currentPage++; displayPage(); } });

        pageLayout->addWidget(btnPrevPage);
        pageLayout->addWidget(lblPageInfo);
        pageLayout->addWidget(btnNextPage);

        rightPanel->addLayout(pageLayout);
        layout->addLayout(rightPanel, 2);
    }

    void retranslateUI()
    {
        dropLabel->setText(t("drop_lbl"));
        sldLabel->setText(t("slider_lbl").arg(sld->value()));
        chkGroup->setText(t("chk_group"));
        btnSearch->setText(t("btn_search"));
        logTitleLbl->setText(t("log_title"));

        fileMenu->setTitle(t("menu_file"));
        actUpdateDb->setText(t("menu_update_db"));
        actExit->setText(t("menu_exit"));

        settingsMenu->setTitle(t("menu_settings"));
        themeSub->setTitle(t("menu_theme"));
        helpMenu->setTitle(t("menu_help"));

        btnPrevPage->setText(t("btn_back"));
        btnNextPage->setText(t("btn_next"));

        setWindowTitle(QString("FaceFin v0.9.8b - [%1]").arg(currentLang == UKRAINIAN ? "UA" : "EN"));

        if (m_currentResults.empty())
            lblPageInfo->setText(t("page_lbl").arg(0).arg(0));
        else
            displayPage();
    }

    bool eventFilter(QObject *obj, QEvent *event) override
    {
        if (obj == dropLabel)
        {
            if (event->type() == QEvent::MouseButtonPress)
            {
                QString fileFilter = "Документи та Фото (*.jpg *.png *.jpeg *.webp *.bmp *.tiff *.pdf *.docx)";
                QString path = QFileDialog::getOpenFileName(this, "Виберіть файл", "", fileFilter);
                if (!path.isEmpty())
                {
                    if (path.endsWith(".pdf", Qt::CaseInsensitive))
                        processPdf(path);
                    else if (path.endsWith(".docx", Qt::CaseInsensitive))
                        processDocx(path);
                    else
                        processImage(path);
                }
                return true;
            }
            else if (event->type() == QEvent::DragEnter)
            {
                QDragEnterEvent *dEvent = static_cast<QDragEnterEvent *>(event);
                if (dEvent->mimeData()->hasUrls())
                    dEvent->acceptProposedAction();
                return true;
            }
            else if (event->type() == QEvent::Drop)
            {
                QDropEvent *dEvent = static_cast<QDropEvent *>(event);
                QString path = dEvent->mimeData()->urls().first().toLocalFile();
                if (!path.isEmpty())
                {
                    if (path.endsWith(".pdf", Qt::CaseInsensitive))
                        processPdf(path);
                    else if (path.endsWith(".docx", Qt::CaseInsensitive))
                        processDocx(path);
                    else
                        processImage(path);
                }
                return true;
            }
        }
        return QMainWindow::eventFilter(obj, event);
    }

    void processImage(const QString &path)
    {
        cv::Mat img = loadMatUtf8(path);
        if (img.empty())
        {
            QMessageBox::warning(this, "Помилка", "Неможливо прочитати файл!");
            return;
        }

        std::vector<FaceInfo> faces = engine->detectFaces(img, 0.5f);
        if (faces.empty())
        {
            QMessageBox::warning(this, "Увага", "Облич на фото оригіналу не знайдено!");
            return;
        }

        targetPath = path;
        int chosenIndex = 0;

        if (faces.size() > 1)
        {
            std::vector<cv::Rect2f> rects;
            for (const auto &f : faces)
                rects.push_back(f.bbox);
            FaceSelectionDialog dlg(path, rects, this);
            if (dlg.exec() == QDialog::Accepted && dlg.selectedIndex != -1)
                chosenIndex = dlg.selectedIndex;
            else
                return;
        }
        else
        {
            selectedEnc = faces[0].embedding;
            selectedLoc = faces[0].bbox;
        }

        if (engine->extractEmbedding(img, faces[chosenIndex]))
        {
            selectedEnc = faces[chosenIndex].embedding;
            selectedLoc = faces[chosenIndex].bbox;
            btnSearch->setEnabled(true);

            cv::Rect bbox(selectedLoc.x, selectedLoc.y, selectedLoc.width, selectedLoc.height);
            cv::Mat crop = img(bbox & cv::Rect(0, 0, img.cols, img.rows)).clone();
            m_targetFaceCrop = crop;

            dropLabel->setPixmap(cvMatToQPixmap(crop).scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
            dropLabel->setText("");
            logConsole->append("<font color='#88c0d0'>[СКАНЕР]</font> Фото успішно підготовлено до пошуку.");
        }
        else
        {
            QMessageBox::warning(this, "Помилка", "Не вдалося згенерувати вектор обличчя!");
        }
    }

    void processDocx(const QString &path)
    {
        logConsole->append("<font color='#88c0d0'>[СКАНЕР]</font> Читання DOCX для пошуку...");
        QApplication::processEvents();

        std::vector<cv::Mat> faceCrops;
        std::vector<std::vector<float>> faceEmbeddings;

        int err = 0;
        zip_t *archive = zip_open(path.toUtf8().constData(), 0, &err);
        if (!archive)
        {
            QMessageBox::warning(this, "Помилка", "Не вдалося відкрити DOCX!");
            return;
        }

        zip_int64_t num_entries = zip_get_num_entries(archive, 0);
        for (zip_int64_t i = 0; i < num_entries; i++)
        {
            const char *name = zip_get_name(archive, i, 0);
            if (!name)
                continue;
            QString fileName(name);
            if (fileName.startsWith("word/media/", Qt::CaseInsensitive))
            {
                zip_stat_t sb;
                if (zip_stat_index(archive, i, 0, &sb) == 0)
                {
                    zip_file_t *f = zip_fopen_index(archive, i, 0);
                    if (f)
                    {
                        QByteArray data;
                        data.resize(sb.size);
                        zip_fread(f, data.data(), sb.size);
                        zip_fclose(f);
                        QImage qimg = QImage::fromData(data);
                        if (!qimg.isNull())
                        {
                            qimg = qimg.convertToFormat(QImage::Format_RGB888);
                            cv::Mat img(qimg.height(), qimg.width(), CV_8UC3, (void *)qimg.bits(), qimg.bytesPerLine());
                            cv::Mat bgrImg;
                            cv::cvtColor(img, bgrImg, cv::COLOR_RGB2BGR);
                            if (!bgrImg.empty())
                            {
                                auto faces = engine->detectFaces(bgrImg, 0.5f);
                                for (auto &face : faces)
                                {
                                    if (engine->extractEmbedding(bgrImg, face))
                                    {
                                        cv::Rect bbox(face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height);
                                        cv::Mat crop = bgrImg(bbox & cv::Rect(0, 0, bgrImg.cols, bgrImg.rows)).clone();
                                        if (!crop.empty())
                                        {
                                            faceCrops.push_back(crop);
                                            faceEmbeddings.push_back(face.embedding);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        zip_close(archive);

        if (faceCrops.empty())
        {
            QMessageBox::warning(this, "Увага", "Облич у цьому DOCX-документі не знайдено!");
            return;
        }

        int chosenIndex = 0;
        if (faceCrops.size() > 1)
        {
            PdfFaceSelectionDialog dlg(faceCrops, this);
            if (dlg.exec() == QDialog::Accepted && dlg.selectedIndex != -1)
                chosenIndex = dlg.selectedIndex;
            else
                return;
        }

        selectedEnc = faceEmbeddings[chosenIndex];
        targetPath = path;
        selectedLoc = cv::Rect2f(0, 0, 0, 0);
        m_targetFaceCrop = faceCrops[chosenIndex];
        btnSearch->setEnabled(true);
        dropLabel->setPixmap(cvMatToQPixmap(faceCrops[chosenIndex]).scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        dropLabel->setText("");
        logConsole->append(QString("<font color='#88c0d0'>[СКАНЕР]</font> Знайдено %1 облич у DOCX.").arg(faceCrops.size()));
    }

    void processPdf(const QString &path)
    {
        logConsole->append("<font color='#88c0d0'>[СКАНЕР]</font> Аналіз PDF...");
        QApplication::processEvents();

        std::vector<cv::Mat> faceCrops;
        std::vector<std::vector<float>> faceEmbeddings;

        FPDF_DOCUMENT doc = FPDF_LoadDocument(path.toUtf8().constData(), nullptr);
        if (!doc)
        {
            logConsole->append(QString("<font color='red'>[ПОМИЛКА PDF]</font> Файл пошкоджений або зашифрований: <b>%1</b>.").arg(QFileInfo(path).fileName()));
            return;
        }

        int pageCount = FPDF_GetPageCount(doc);
        for (int i = 0; i < pageCount; ++i)
        {
            FPDF_PAGE page = FPDF_LoadPage(doc, i);
            if (!page)
            {
                logConsole->append(QString("<font color='orange'>[УВАГА]</font> Пошкоджена сторінка %1.").arg(i + 1));
                continue;
            }

            double widthPoints = FPDF_GetPageWidth(page);
            double heightPoints = FPDF_GetPageHeight(page);
            float scale = 150.0f / 72.0f;
            int width = static_cast<int>(widthPoints * scale);
            int height = static_cast<int>(heightPoints * scale);

            cv::Mat pageBGRA(height, width, CV_8UC4, cv::Scalar(255, 255, 255, 255));
            FPDF_BITMAP bitmap = FPDFBitmap_CreateEx(width, height, FPDFBitmap_BGRA, pageBGRA.data, static_cast<int>(pageBGRA.step));
            if (!bitmap)
            {
                FPDF_ClosePage(page);
                continue;
            }

            FPDF_RenderPageBitmap(bitmap, page, 0, 0, width, height, 0, 0);
            cv::Mat bgrImg;
            cv::cvtColor(pageBGRA, bgrImg, cv::COLOR_BGRA2BGR);

            if (!bgrImg.empty())
            {
                auto faces = engine->detectFaces(bgrImg, 0.5f);
                for (auto &face : faces)
                {
                    if (engine->extractEmbedding(bgrImg, face))
                    {
                        cv::Rect bbox(face.bbox.x, face.bbox.y, face.bbox.width, face.bbox.height);
                        cv::Mat crop = bgrImg(bbox & cv::Rect(0, 0, bgrImg.cols, bgrImg.rows)).clone();
                        if (!crop.empty())
                        {
                            faceCrops.push_back(crop);
                            faceEmbeddings.push_back(face.embedding);
                        }
                    }
                }
            }
            FPDFBitmap_Destroy(bitmap);
            FPDF_ClosePage(page);
        }
        FPDF_CloseDocument(doc);

        if (faceCrops.empty())
        {
            QMessageBox::warning(this, "Увага", "Облич у цьому PDF-документі не знайдено!");
            return;
        }

        int chosenIndex = 0;
        if (faceCrops.size() > 1)
        {
            PdfFaceSelectionDialog dlg(faceCrops, this);
            if (dlg.exec() == QDialog::Accepted && dlg.selectedIndex != -1)
                chosenIndex = dlg.selectedIndex;
            else
                return;
        }

        selectedEnc = faceEmbeddings[chosenIndex];
        targetPath = path;
        selectedLoc = cv::Rect2f(0, 0, 0, 0);
        m_targetFaceCrop = faceCrops[chosenIndex];
        btnSearch->setEnabled(true);
        dropLabel->setPixmap(cvMatToQPixmap(faceCrops[chosenIndex]).scaled(200, 200, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        dropLabel->setText("");
        logConsole->append(QString("<font color='#88c0d0'>[ІНФО]</font> Знайдено %1 облич у PDF.").arg(faceCrops.size()));
    }

    void runSearch()
    {
        btnSearch->setEnabled(false);
        float threshold = sld->value() / 100.0f;
        logConsole->append(QString("<font color='#bd93f9'>[ПОШУК]</font> Пошук по індексу... (Поріг: %1%)").arg(sld->value()));
        QApplication::processEvents();

        m_currentResults = db->search(selectedEnc, threshold, 200);

        if (chkGroup->isChecked() && !m_currentResults.empty())
        {
            std::vector<SearchResult> groupedResults;
            std::vector<std::vector<float>> groupedEmbeddings;

            for (auto &res : m_currentResults)
            {
                std::vector<float> resEmb = db->getEmbeddingById(res.id);
                if (resEmb.empty())
                    continue;

                bool foundGroup = false;
                for (size_t i = 0; i < groupedResults.size(); ++i)
                {
                    float ip = 0.0f;
                    for (size_t j = 0; j < 512; ++j)
                        ip += resEmb[j] * groupedEmbeddings[i][j];

                    if (ip > 0.98f)
                    {
                        if (std::find(groupedResults[i].duplicatePaths.begin(), groupedResults[i].duplicatePaths.end(), res.filePath) == groupedResults[i].duplicatePaths.end() && groupedResults[i].filePath != res.filePath)
                        {
                            groupedResults[i].duplicatePaths.push_back(res.filePath);
                        }
                        foundGroup = true;
                        break;
                    }
                }
                if (!foundGroup)
                {
                    groupedResults.push_back(res);
                    groupedEmbeddings.push_back(resEmb);
                }
            }
            m_currentResults = groupedResults;
        }

        m_currentPage = 0;

        if (m_currentResults.empty())
        {
            logConsole->append("<font color='red'>[ПОШУК]</font> Жодного схожого обличчя не знайдено.");
            displayPage();
            btnSearch->setEnabled(true);
            return;
        }

        logConsole->append(QString("<font color='green'>[ПОШУК]</font> Успішно знайдено: <b>%1</b>").arg(m_currentResults.size()));
        displayPage();
        btnSearch->setEnabled(true);
    }

    void displayPage()
    {
        QLayoutItem *child;
        while ((child = resGrid->takeAt(0)) != nullptr)
        {
            if (child->widget())
                delete child->widget();
            delete child;
        }

        if (m_currentResults.empty())
        {
            lblPageInfo->setText(t("page_lbl").arg(0).arg(0));
            btnPrevPage->setEnabled(false);
            btnNextPage->setEnabled(false);
            return;
        }

        int totalPages = (m_currentResults.size() + RESULTS_PER_PAGE - 1) / RESULTS_PER_PAGE;
        lblPageInfo->setText(t("page_lbl").arg(m_currentPage + 1).arg(totalPages));

        btnPrevPage->setEnabled(m_currentPage > 0);
        btnNextPage->setEnabled(m_currentPage < totalPages - 1);

        int startIdx = m_currentPage * RESULTS_PER_PAGE;
        int endIdx = std::min(startIdx + RESULTS_PER_PAGE, (int)m_currentResults.size());

        int r = 0, c = 0;
        for (int i = startIdx; i < endIdx; ++i)
        {
            resGrid->addWidget(createResultCard(m_currentResults[i]), r, c);
            if (++c >= 4)
            {
                c = 0;
                r++;
            }
        }
    }

    QWidget *createResultCard(const SearchResult &res)
    {
        QWidget *card = new QWidget();
        card->setFixedSize(180, 240);
        card->setStyleSheet("border-radius: 5px; border: 1px solid rgba(150, 150, 150, 0.4);");
        QVBoxLayout *lay = new QVBoxLayout(card);

        QLabel *imgLbl = new QLabel();
        QImage resImg;
        QString relOrAbsPath = QString::fromStdString(res.cachePath);
        QString absCache = QDir(QCoreApplication::applicationDirPath()).absoluteFilePath(relOrAbsPath);

        if (!res.cachePath.empty() && QFile::exists(absCache))
            resImg.load(absCache);
        else if (!res.cachePath.empty() && QFile::exists(relOrAbsPath))
            resImg.load(relOrAbsPath);
        else
            resImg.load(QString::fromStdString(res.filePath));

        if (!resImg.isNull())
            imgLbl->setPixmap(QPixmap::fromImage(resImg).scaled(160, 160, Qt::KeepAspectRatio, Qt::SmoothTransformation));
        else
            imgLbl->setText("НЕМАЄ ФОТО\nАБО PDF");

        imgLbl->setMinimumSize(160, 160);
        imgLbl->setAlignment(Qt::AlignCenter);

        QString displayName = res.name.empty() ? QFileInfo(QString::fromStdString(res.filePath)).fileName() : QString::fromStdString(res.name);
        QString dupInfo = res.duplicatePaths.empty() ? "" : QString("<br><span style='color:#e3a828; font-size: 11px;'>[+%1 копій]</span>").arg(res.duplicatePaths.size());
        QLabel *infoLbl = new QLabel(QString("<b style='color:#4caf50;'>%1%</b>%2<br>%3").arg(res.similarity * 100.0f, 0, 'f', 1).arg(dupInfo).arg(displayName));
        infoLbl->setWordWrap(true);
        infoLbl->setStyleSheet("font-size: 11px; border: none;");

        lay->addWidget(imgLbl);
        lay->addWidget(infoLbl);

        card->setContextMenuPolicy(Qt::CustomContextMenu);
        connect(card, &QWidget::customContextMenuRequested, [this, res, infoLbl](const QPoint &pos)
                {
            Q_UNUSED(pos); QMenu menu;
            QAction* actName = menu.addAction("✏️ Додати ім'я");
            QAction* actComp = menu.addAction("🔍 Порівняти фото");
            
            QAction* actOpenSingle = nullptr; QMenu* openMenu = nullptr;

            if (res.duplicatePaths.empty()) {
                actOpenSingle = menu.addAction("📁 Відкрити папку");
            } else {
                openMenu = menu.addMenu("📁 Відкрити папку...");
                QAction* actMain = openMenu->addAction("Головний файл: " + QFileInfo(QString::fromStdString(res.filePath)).fileName());
                connect(actMain, &QAction::triggered, [res](){ QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(QString::fromStdString(res.filePath)).absolutePath())); });
                for (size_t i = 0; i < res.duplicatePaths.size(); ++i) {
                    QString dupPath = QString::fromStdString(res.duplicatePaths[i]);
                    QAction* actDup = openMenu->addAction(QString("Копія %1: %2").arg(i+1).arg(QFileInfo(dupPath).fileName()));
                    connect(actDup, &QAction::triggered, [dupPath](){ QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(dupPath).absolutePath())); });
                }
            }

            QAction* sel = menu.exec(QCursor::pos());

            if (sel && sel == actOpenSingle) {
                QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(QString::fromStdString(res.filePath)).absolutePath()));
            } else if (sel == actComp) {
                ComparisonDialog dlg(targetPath, m_targetFaceCrop, res, db, this); dlg.exec();
            } else if (sel == actName) {
                bool ok; QString text = QInputDialog::getText(this, "Додати ім'я", "Введіть ім'я особи в БД:", QLineEdit::Normal, QString::fromStdString(res.name), &ok);
                if (ok) {
                    db->updateName(res.id, text.toStdString());
                    QString dupInfoNew = res.duplicatePaths.empty() ? "" : QString("<br><span style='color:#e3a828; font-size: 11px;'>[+%1 копій]</span>").arg(res.duplicatePaths.size());
                    QString newTitle = text.isEmpty() ? QFileInfo(QString::fromStdString(res.filePath)).fileName() : text;
                    infoLbl->setText(QString("<b style='color:#4caf50;'>%1%</b>%2<br>%3").arg(res.similarity * 100.0f, 0, 'f', 1).arg(dupInfoNew).arg(newTitle));
                }
            } });
        return card;
    }

    void updateDB()
    {
        if (workerThread && workerThread->isRunning())
        {
            QMessageBox::warning(this, "Увага", "Процес сканування вже запущено у фоні!");
            return;
        }

        MultiFolderDialog dlg(this);
        if (dlg.exec() != QDialog::Accepted)
            return;

        QStringList dirPaths = dlg.getSelectedFolders();
        if (dirPaths.isEmpty())
            return;

        scanProgress->setValue(0);
        scanProgress->show();
        menuBar()->setEnabled(false);
        workerThread = new QThread(this);
        worker = new DBWorker(dirPaths, engine, db, QApplication::applicationDirPath());
        worker->moveToThread(workerThread);

        connect(workerThread, &QThread::started, worker, &DBWorker::process);
        connect(worker, &DBWorker::logMsg, logConsole, &QTextBrowser::append);
        connect(worker, &DBWorker::progressUpdate, this, [this](int current, int total)
                { scanProgress->setMaximum(total); scanProgress->setValue(current); });
        connect(worker, &DBWorker::finishedScan, this, [this](int normalCount, int enhancedCount)
                {
            scanProgress->hide(); menuBar()->setEnabled(true); int total = normalCount + enhancedCount;
            db->saveIndex(); 
            logConsole->append(QString("<font color='#4caf50'>[СИСТЕМА]</font> Всього у базі: <b>%1</b>").arg(db->getRecordCount()));
            QString msgBoxText = QString("<h3>🎉 Сканування завершено!</h3><p>Базу даних успішно оновлено.</p><ul><li>Знайдено: <b>%1</b></li><li>Виявлено після покращення якості: <b style='color:#e3a828;'>%2</b></li></ul><p>Всього додано нових біометричних відбитків: <b>%3</b></p>").arg(normalCount).arg(enhancedCount).arg(total);
            QMessageBox msgBox(this); msgBox.setWindowTitle("Готово"); msgBox.setTextFormat(Qt::RichText); msgBox.setText(msgBoxText); msgBox.exec();
            workerThread->quit(); workerThread->wait();
            worker->deleteLater(); workerThread->deleteLater(); workerThread = nullptr; });

        workerThread->start();
    }
};

int main(int argc, char *argv[])
{
    // Ініціалізуємо движок Google PDFium глобально для всієї програми (найбезпечніший метод)
    FPDF_InitLibrary();

    QApplication app(argc, argv);
    FaceFinderApp window;
    window.show();

    int result = app.exec();

    // Очищаємо пам'ять при закритті
    FPDF_DestroyLibrary();
    return result;
}

#include "main.moc"
