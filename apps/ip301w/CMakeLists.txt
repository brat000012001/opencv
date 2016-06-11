set(OPENCV_IPW_DEPS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_video opencv_videoio)
ocv_check_dependencies(${OPENCV_IPW_DEPS})

if(NOT OCV_DEPENDENCIES_FOUND)
  return()
endif()

project(ip301w)
set(the_target opencv_ip301w)

ocv_target_include_directories(${the_target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${OpenCV_SOURCE_DIR}/include/opencv")
ocv_target_include_modules_recurse(${the_target} ${OPENCV_IPW_DEPS})

file(GLOB SRCS *.cpp)

set(ipw_files ${SRCS} ${HDRS})

ocv_add_executable(${the_target} ${ipw_files})
ocv_target_link_libraries(${the_target} ${OPENCV_IPW_DEPS})

set_target_properties(${the_target} PROPERTIES
                      DEBUG_POSTFIX "${OPENCV_DEBUG_POSTFIX}"
                      ARCHIVE_OUTPUT_DIRECTORY ${LIBRARY_OUTPUT_PATH}
                      RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH}
                      OUTPUT_NAME "opencv_ip301w")

if(ENABLE_SOLUTION_FOLDERS)
  set_target_properties(${the_target} PROPERTIES FOLDER "applications")
endif()

if(INSTALL_CREATE_DISTRIB)
  if(BUILD_SHARED_LIBS)
    install(TARGETS ${the_target} RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} CONFIGURATIONS Release COMPONENT dev)
  endif()
else()
  install(TARGETS ${the_target} OPTIONAL RUNTIME DESTINATION ${OPENCV_BIN_INSTALL_PATH} COMPONENT dev)
endif()
