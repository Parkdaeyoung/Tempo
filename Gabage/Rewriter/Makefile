CXX = g++
CFLAGS = -fno-rtti -std=c++11 -Wall

LLVM_SRC_PATH = 
LLVM_BUILD_PATH = /usr/local/include/clang

LLVM_BIN_PATH = $(LLVM_BUILD_PATH)/Debug+Asserts/bin
LLVM_LIBS=core mc
LLVM_CONFIG_COMMAND = $(LLVM_BIN_PATH)/llvm-config --cxxflags --ldflags \
					  --libs $(LLVM_LIBS)
CLANG_BUILD_FLAGS = -I$(LLVM_SRC_PATH)/tools/clang/include \
					-I$(LLVM_BUILD_PATH)/tools/clang/include

CLANGLIBS = \
			-lclangFrontendTool -lclangFrontend -lclangDriver \
			-lclangSerialization -lclangCodeGen -lclangParse \
			-lclangSema -lclangStaticAnalyzerFrontend \
			-lclangStaticAnalyzerCheckers -lclangStaticAnalyzerCore \
			-lclangAnalysis -lclangARCMigrate -lclangRewrite \
			-lclangEdit -lclangAST -lclangLex -lclangBasic\
			-lclangTooling\
		   	$(shell llvm-config --libs)\
			$(shell llvm-config --system-libs)

all: main

main: main.cpp
	$(CXX) $^ $(CFLAGS) -o $@ $(CLANG_BUILD_FLAGS) $(CLANGLIBS)

clean: 
	rm -rf *.o *.ll rewritersample transformer
