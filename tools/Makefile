ARCH ?= sm_90a
TARGETS ?= gpc_query gpc_query_matrix

all: $(TARGETS)

%: %.cu
	nvcc -arch=$(ARCH) -O2 -std=c++17 -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: clean
