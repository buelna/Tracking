UTILS_DIR = ../utils
INCLUDE_DIR = ../includes
EXTERN_DIR = ../extern

test = tracking_test

objects = $(UTILS_DIR)/tracking_utils.o $(UTILS_DIR)/matrix_operations.o $(UTILS_DIR)/fft_utils.o 
includes = $(INCLUDE_DIR)/tracking_utils.h $(INCLUDE_DIR)/matrix_operations.h $(INCLUDE_DIR)/fft_utils.h
extern = $(EXTERN_DIR)/liblapack.so.3.0

all : $(test)

tracking_test : tracking.c $(objects) $(extern) $(includes)
	gcc -ggdb `pkg-config --cflags opencv` -o tracking tracking.c $(objects) $(extern) -lm -lfftw3 -std=c99 `pkg-config --libs opencv`

$(UTILS_DIR)/fft_utils.o :$(UTILS_DIR)/fft_utils.c $(includes)
	@cd $(UTILS_DIR) ; gcc -c fft_utils.c -lfftw3 -lm -std=c99
$(UTILS_DIR)/tracking_utils.o :$(UTILS_DIR)/tracking_utils.c $(includes)
	@cd $(UTILS_DIR) ; gcc -c tracking_utils.c -std=c99
$(UTILS_DIR)/matrix_operations.o :$(UTILS_DIR)/matrix_operations.c $(includes)
	@cd $(UTILS_DIR) ; gcc -c matrix_operations.c -std=c99




.PHONY : clean

clean :
	-rm tracking $(objects) *~ $(UTILS_DIR)/*~
