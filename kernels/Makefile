BIN     := bin/ostir_kernel
SRC     := ostir_kernel.c
CFLAGS  := -O3 -std=c11 -Wall -Wextra

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
  CFLAGS += -march=native -mtune=native
else
  CFLAGS += -mcpu=native
endif

LDLIBS  := -lm

.PHONY: all clean
all: $(BIN)

$(BIN): $(SRC) | bin
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

bin:
	mkdir -p bin

clean:
	rm -rf bin

Q4BIN := bin/q4_gemv

.PHONY: q4
q4: $(Q4BIN)

$(Q4BIN): q4_gemv.c | bin
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)
