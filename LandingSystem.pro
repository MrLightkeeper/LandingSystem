SOURCES += \
    main.cpp
INCLUDEPATH += /usr/local/include/opencv2

HEADERS += \
    main.h

LIBS += -L"/usr/local/lib"
LIBS += -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio
