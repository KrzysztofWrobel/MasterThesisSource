cmake_minimum_required(VERSION 2.8)
project( 3Dreconstruct )

#This one is needed in order to find cvsba
SET(CMAKE_MODULE_PATH ${CMAKE_INSTALL_PREFIX}/lib/cmake/ )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
configure_file(out_camera_data.yml out_camera_data.yml COPYONLY)

find_package(PCL 1.7 REQUIRED)
find_package(GLEW 1.10 REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
link_directories(${GLEW_LIBRARY_DIRS})

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


FIND_PACKAGE( Boost 1.55 COMPONENTS program_options REQUIRED )
INCLUDE_DIRECTORIES( ${Boost_INCLUDE_DIR} )
IF(APPLE)
    FIND_LIBRARY(Accelerate_LIBRARY Accelerate REQUIRED)
ENDIF(APPLE)

find_package( OpenCV 2.4.10 REQUIRED )
find_package( cvsba REQUIRED )

set(SOURCES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SOURCES ${SOURCES_DIR}/IOUtils.h ${SOURCES_DIR}/TestUtils.h ${SOURCES_DIR}/Utils64.h ${SOURCES_DIR}/FeatureMatching.h ${SOURCES_DIR}/MultiView.h ${SOURCES_DIR}/modelest.h
            ${SOURCES_DIR}/FeatureMatching.cpp ${SOURCES_DIR}/MultiView.cpp ${SOURCES_DIR}/modelest.cpp ${SOURCES_DIR}/5point.cpp ${SOURCES_DIR}/Rpoly.cpp ${SOURCES_DIR}/TestUtils.cpp)
set(HEADER_FILES ${SOURCES_DIR}/IOUtils.h ${SOURCES_DIR}/TestUtils.h ${SOURCES_DIR}/Utils64.h ${SOURCES_DIR}/FeatureMatching.h ${SOURCES_DIR}/MultiView.h ${SOURCES_DIR}/modelest.h ${SOURCES_DIR}/levmar.h ${SOURCES_DIR}/5point.h ${SOURCES_DIR}/Rpoly.h ${SOURCES_DIR}/Polynomial.h ${SOURCES_DIR}/Mblock.h )
set(LEVMAR_LIB ${SOURCES_DIR}/liblevmar.a)
INCLUDE_DIRECTORIES( ${SOURCES} )

add_executable( 5point src/main5p.cpp ${HEADER_FILES} ${SOURCES})
add_executable( ReconstructVideo src/main1.cpp )
add_executable( ReconstructImagesEnhanced src/main2.cpp ${HEADER_FILES} ${SOURCES})
add_executable( test_Reconstruct src/test_Reconstruct.cpp ${HEADER_FILES} ${SOURCES})
add_executable( Efficiency src/testOfImplementationEfficency.cpp ${HEADER_FILES} ${SOURCES})
add_executable( test_bundleAdjust src/test_bundleAdjust.cpp)
add_executable( cvsba_simple_test src/cvsba_simple_test.cpp)
add_executable( levmar src/levmar.h src/expfit.c)

target_link_libraries( 5point ${OpenCV_LIBS} )
target_link_libraries( ReconstructVideo ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES} )
target_link_libraries( ReconstructImagesEnhanced ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES} )
target_link_libraries( test_Reconstruct ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES}  five-point-nister ${cvsba_LIBS})
target_link_libraries( Efficiency ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PCL_LIBRARIES} ${LEVMAR_LIB} ${Accelerate_LIBRARY} five-point-nister )
target_link_libraries( test_bundleAdjust ${PCL_LIBRARIES}  ${OpenCV_LIBS} ${cvsba_LIBS} )
target_link_libraries( cvsba_simple_test ${OpenCV_LIBS} ${cvsba_LIBS} )
target_link_libraries( levmar ${LEVMAR_LIB} ${Accelerate_LIBRARY} )

add_subdirectory(src/four-point-numerical)
add_subdirectory(src/four-point-groebner)
add_subdirectory(src/five-point-nister)
add_subdirectory(src/one-point)

add_executable(demo src/demo.cpp)
target_link_libraries(demo five-point-nister four-point-numerical four-point-groebner ${OpenCV_LIBS})
