# Makefile — CUDA-ready; works now (no .cu yet) and later when you add kernels

CXX      := g++
NVCC     := nvcc

INCLUDE  := -Iinclude
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic $(INCLUDE)
# Arch flags come later when you add CUDA kernels; keep simple for now:
NVCCFLAGS := -O2 -Xcompiler -Wall,-Wextra,-pedantic $(INCLUDE)

TARGET   := gpu_pipeline

# Sources
CPPSRCS  := $(wildcard src/*.cpp)
CUSRC    := $(wildcard src/*.cu)

# Objects
OBJDIR   := build
OBJ_CPU  := $(CPPSRCS:src/%.cpp=$(OBJDIR)/%.o)
OBJ_CUDA := $(CUSRC:src/%.cu=$(OBJDIR)/%.cu.o)
OBJS     := $(OBJ_CPU) $(OBJ_CUDA)

# Libraries (std::filesystem on older GCC needs this; if link fails, uncomment):
# LIBS += -lstdc++fs

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) -o $@ $(OBJS) $(LIBS)

# Compile C++ sources
$(OBJDIR)/%.o: src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources (used once you add .cu files)
$(OBJDIR)/%.cu.o: src/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET) --help

clean:
	rm -rf $(OBJDIR) $(TARGET)
