# Makefile for Chassis FFI C examples

CC = gcc
CFLAGS = -Wall -Wextra -O2 -I../include
LDFLAGS = -L../../target/release -lchassis_ffi -lm

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    LIB_PATH = LD_LIBRARY_PATH=../../target/release
    LIB_EXT = .so
endif
ifeq ($(UNAME_S),Darwin)
    LIB_PATH = DYLD_LIBRARY_PATH=../../target/release
    LIB_EXT = .dylib
endif

all: example

# Build the Rust library first
lib:
	cd ../.. && cargo build --release --package chassis-ffi

# Build the C example
example: example.c lib
	$(CC) $(CFLAGS) -o example example.c $(LDFLAGS)

# Run the example
run: example
	$(LIB_PATH) ./example

# Clean build artifacts
clean:
	rm -f example
	rm -f *.chassis
	rm -f *.o

# Clean everything including Rust artifacts
distclean: clean
	cd ../.. && cargo clean

.PHONY: all lib run clean distclean
