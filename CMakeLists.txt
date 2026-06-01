cmake_minimum_required(VERSION 3.20)
project(FaceFin VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# --- МІНІМАЛІСТИЧНИЙ ПОШУК QT5 ---
list(APPEND CMAKE_PREFIX_PATH "E:/DEV/vcpkg/installed/x64-windows")
find_package(Qt5 COMPONENTS Widgets Core Gui REQUIRED)

find_package(OpenCV REQUIRED)
find_package(unofficial-sqlite3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(onnxruntime CONFIG REQUIRED)
find_package(faiss CONFIG REQUIRED)
find_package(libzip CONFIG REQUIRED)


set(SOURCES
    src/main.cpp 
    src/core/FaceEngine.cpp
    src/core/Database.cpp
)

list(APPEND SOURCES appicon.rc)
add_executable(FaceFin WIN32 ${SOURCES})

target_include_directories(FaceFin PRIVATE "${CMAKE_SOURCE_DIR}/third_party/pdfium/include")

target_link_libraries(FaceFin PRIVATE
    Qt5::Widgets
    Qt5::Core
    Qt5::Gui
    ${OpenCV_LIBS}
    unofficial::sqlite3::sqlite3
    nlohmann_json::nlohmann_json
    onnxruntime::onnxruntime
    faiss
    libzip::zip
    "${CMAKE_SOURCE_DIR}/third_party/pdfium/lib/pdfium.dll.lib"
)

# ==========================================
# АВТОМАТИЧНЕ ПАКУВАННЯ У PORTABLE-ВЕРСІЮ
# ==========================================

# 1. Отримуємо шляхи до папок vcpkg
get_target_property(QT_CORE_LOCATION Qt5::Core LOCATION)
get_filename_component(VCPKG_BIN_DIR "${QT_CORE_LOCATION}" DIRECTORY)
get_filename_component(VCPKG_ROOT_DIR "${VCPKG_BIN_DIR}" DIRECTORY)

# Збираємо список АБСОЛЮТНО ВСІХ .dll файлів, які vcpkg скомпілював для цього проекту
file(GLOB VCPKG_ALL_DLLS "${VCPKG_BIN_DIR}/*.dll")

# 2. ЖОРСТКЕ КОПІЮВАННЯ ВСІХ DLL (Це на 100% вирішує проблему з harfbuzz, pcre, png і т.д.)
add_custom_command(TARGET FaceFin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        ${VCPKG_ALL_DLLS} "$<TARGET_FILE_DIR:FaceFin>"
    COMMENT ">>> 100% Гарантія: Копіювання ВСІХ бібліотек з vcpkg..."
)

# 3. Копіюємо папки плагінів Qt (Без них не буде інтерфейсу)
add_custom_command(TARGET FaceFin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        "${VCPKG_ROOT_DIR}/plugins/platforms" "$<TARGET_FILE_DIR:FaceFin>/platforms"
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        "${VCPKG_ROOT_DIR}/plugins/imageformats" "$<TARGET_FILE_DIR:FaceFin>/imageformats"
    COMMENT ">>> Копіювання плагінів Qt5 (platforms, imageformats)..."
)

# 4. Копіюємо ресурси програми
add_custom_command(TARGET FaceFin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        "${CMAKE_SOURCE_DIR}/models" "$<TARGET_FILE_DIR:FaceFin>/models"
    COMMAND ${CMAKE_COMMAND} -E copy_directory 
        "${CMAKE_SOURCE_DIR}/tools" "$<TARGET_FILE_DIR:FaceFin>/tools"
    COMMENT ">>> Копіювання моделей та утиліт..."
)

# 5. АВТОМАТИЧНЕ КОПІЮВАННЯ СИСТЕМНИХ БІБЛІОТЕК MSVC (msvcp140.dll, vcruntime140.dll)
set(CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS_SKIP TRUE)
include(InstallRequiredSystemLibraries)

foreach(SYSTEM_LIB ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
    add_custom_command(TARGET FaceFin POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${SYSTEM_LIB}"
        "$<TARGET_FILE_DIR:FaceFin>"
        COMMENT ">>> Копіювання системної DLL: ${SYSTEM_LIB}"
    )

# 6. СТВОРЮЄМО qt.conf ДЛЯ ПРАВИЛЬНОГО ЗАВАНТАЖЕННЯ ПЛАГІНІВ (JPEG, PNG, TIFF)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/qt.conf" "[Paths]\nPrefix=.\nPlugins=.\n")
add_custom_command(TARGET FaceFin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_BINARY_DIR}/qt.conf"
        "$<TARGET_FILE_DIR:FaceFin>/qt.conf"
    COMMENT ">>> Створення qt.conf для прив'язки плагінів..."
)
# 7. Автоматично копіюємо pdfium.dll до нашого FaceFin.exe при компіляції
add_custom_command(TARGET FaceFin POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/third_party/pdfium/bin/pdfium.dll"
        "$<TARGET_FILE_DIR:FaceFin>"
    COMMENT ">>> Копіювання pdfium.dll..."
)    
endforeach()