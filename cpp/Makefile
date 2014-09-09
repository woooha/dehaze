CXX=clang++
CFlags=-Wall -L/usr/local/lib -I/usr/local/include -lgflags

.PHONY: a bc 

dehaze: main.cc dehaze.h dehaze.cc
	$(CXX) $(CFlags) main.cc dehaze.cc /usr/local/lib/libopencv_* -o $@ 

clean:
	rm -rf dehaze
