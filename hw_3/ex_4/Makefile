# From https://x.momo86.net/?p=29

CXX=g++
CXXFLAGS=-std=c++11 -I./include -O3 -g -G -Xcompiler -Wall

NVCC=nvcc
# ARCH = sm_30 # can't be used since I'm using a different architecture (see previous assignment report for details)
ARCH=sm_75
NVCCFLAGS= -I./include -arch=$(ARCH) -std=c++11 -O3 -g -G -Xcompiler -Wall --compiler-bindir=$(CXX)

SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR=src
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

BIN := ./bin
TARGET=sputniPIC.out

all: dir $(BIN)/$(TARGET)

dir: ${BIN}
  
${BIN}:
	mkdir -p $(BIN)

$(BIN)/$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(SRCDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)