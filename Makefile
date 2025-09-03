# Makefile — CUDA-ready; builds C++ and .cu, links with nvcc

CXX      := g++
NVCC     := nvcc

INCLUDE  := -Iinclude
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic $(INCLUDE)

# Target GPU archs: feel free to trim if needed
GPU_ARCHS ?= 60 70 75 80 86
NVCC_ARCH := $(foreach arch,$(GPU_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch))
# NVCCFLAGS := -O2 $(NVCC_ARCH) -Xcompiler -Wall,-Wextra,-pedantic $(INCLUDE)
# NVCCFLAGS := -O2 $(NVCC_ARCH) -Xcompiler -Wall,-Wextra $(INCLUDE)
NVCCFLAGS := -std=c++17 -O2 $(NVCC_ARCH) -Xcompiler -Wall,-Wextra $(INCLUDE)

TARGET   := gpu_pipeline

CPPSRCS  := $(wildcard src/*.cpp)
CUSRC    := $(wildcard src/*.cu)
OBJDIR   := build
OBJ_CPU  := $(CPPSRCS:src/%.cpp=$(OBJDIR)/%.o)
OBJ_CUDA := $(CUSRC:src/%.cu=$(OBJDIR)/%.cu.o)
OBJS     := $(OBJ_CPU) $(OBJ_CUDA)

# If you hit std::filesystem link errors on older toolchains, uncomment:
# LIBS += -lstdc++fs

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ $(OBJS) $(LIBS)

$(OBJDIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/%.cu.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) --help

clean:
	rm -rf $(OBJDIR) $(TARGET)
