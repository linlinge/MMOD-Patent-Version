cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(COR)
set(CMAKE_BUILD_TYPE "Debug")
set(Env{PATH} "/home/llg/anaconda3/pkgs/python-3.7.4-h265db76_1/lib/")

# PCL
find_package(PCL 1.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# OpenMP
FIND_PACKAGE( OpenMP REQUIRED)
if(OPENMP_FOUND)
message("OPENMP FOUND")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("/home/llg/anaconda3/lib/python3.7/site-packages/numpy/core/include/")
# OpenCV 
FIND_PACKAGE(OpenCV REQUIRED)

# Eigen
include_directories("/usr/include/eigen3")

# Ceres
# find_package(Ceres REQUIRED)
# include_directories(${CERES_INCLUDE_DIRS})


############################################# executable #############################################
include_directories(glib .)
ADD_SUBDIRECTORY(glib)
aux_source_directory(. MAIN_SRC)
add_executable (COR main.cpp DataMining.cpp ImprovedLoop.cpp EdgeDetection.cpp AngleBasedOutlier.cpp PointSetFeatures.cpp RunPython.cpp)


###########################################  link library ###############################################
target_link_libraries(COR GLIB)

# PCL
target_link_libraries (COR ${PCL_LIBRARIES})

# OpenCV
target_link_libraries(COR ${OpenCV_LIBS})

# Ceres
# target_link_libraries(pcl_demo ${CERES_LIBRARIES})

# Python
target_link_libraries(COR ${PYTHON_LIBRARIES})
