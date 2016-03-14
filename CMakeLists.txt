cmake_minimum_required(VERSION 3.0)

include(ExternalProject)
include(CheckIncludeFile)

project(detector-markup-tool)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem system REQUIRED)

check_include_file(libswscale/swscale.h HAVE_SWSCALE)
if (HAVE_SWSCALE)
	set(FFMPEG_LIBS avcodec avformat avutil swscale)
	add_definitions(-DHAVE_SWSCALE)
else()
	set(FFMPEG_LIBS avcodec avformat avutil)
endif()

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)

include_directories(${Qt5Concurrent_INCLUDE_DIRS})
include_directories(${Qt5Core_INCLUDE_DIRS})
include_directories(${Qt5Network_INCLUDE_DIRS})
include_directories(${Qt5Widgets_INCLUDE_DIRS})

add_definitions(${Qt5Widgets_DEFINITIONS})
#add_definitions(-DVERSTR="1.0.0")

ExternalProject_Add(
	aifil-lib
	PREFIX aifil-utils
	GIT_REPOSITORY "https://github.com/avtomaton/toolchain-cpp.git"
	GIT_TAG "master"
	INSTALL_COMMAND "")
ExternalProject_Get_property(aifil-lib source_dir)
include_directories(${source_dir})
ExternalProject_Get_property(aifil-lib binary_dir)
link_directories(${binary_dir}/common)
link_directories(${binary_dir}/imgproc)
link_directories(${binary_dir}/video)

ExternalProject_Add(
	anfisa-lib
	PREFIX anfisa-lib
	GIT_REPOSITORY "https://github.com/avtomaton/anfisa.git"
	GIT_TAG "master"
	CMAKE_ARGS -DCMAKE_CXX_FLAGS=-I${source_dir}
	INSTALL_COMMAND "")
ExternalProject_Get_property(anfisa-lib source_dir)
include_directories(${source_dir})
ExternalProject_Get_property(anfisa-lib binary_dir)
link_directories(${binary_dir})

include_directories(${CMAKE_SOURCE_DIR})

set(APP_FORMS
	tool/detector-gui.ui
)

set(APP_SRC
	tool/detector-gui.cpp
	tool/gt-utils.cpp
	tool/dump-utils.cpp
)

#qt5_add_resources(QRC_SOURCES resources.qrc)
qt5_wrap_ui(UI_HEADERS ${APP_FORMS})

add_executable(${PROJECT_NAME}
	${APP_SRC}
	${UI_HEADERS}
	${QRC_SOURCES}
)
add_dependencies(anfisa-lib aifil-lib)
add_dependencies(${PROJECT_NAME} aifil-lib)
add_dependencies(${PROJECT_NAME} anfisa-lib)
qt5_use_modules(${PROJECT_NAME} Core Widgets)
target_link_libraries(${PROJECT_NAME}
	aifil-utils-video
	aifil-utils-imgproc
	aifil-utils-common
	anfisa-minimal
	${FFMPEG_LIBS}
	${Boost_LIBRARIES}
	${OpenCV_LIBS}
)
