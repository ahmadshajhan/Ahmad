# =============================================================================
# Makefile - CNN Framework (Pure C Deep Learning)
#
# Compiles the modular CNN framework with C99 standard.
# No external ML or BLAS libraries -- only the C standard library.
#
# Usage:
#   make          - Build the project
#   make run      - Build and run
#   make clean    - Remove build artifacts
#   make debug    - Build with debug symbols and sanitizers
# =============================================================================

CC       = gcc
CFLAGS   = -std=c99 -Wall -Wextra -Wpedantic -O2
LDFLAGS  = -lm

# Debug build flags
DEBUG_CFLAGS = -std=c99 -Wall -Wextra -Wpedantic -g -O0 -fsanitize=address,undefined

# Directories
SRC_DIR  = src
INC_DIR  = include
BUILD_DIR = build

# Source files
SRCS = $(SRC_DIR)/tensor.c \
       $(SRC_DIR)/conv2d.c \
       $(SRC_DIR)/pooling.c \
       $(SRC_DIR)/activation.c \
       $(SRC_DIR)/dense.c \
       $(SRC_DIR)/flatten.c \
       $(SRC_DIR)/layers.c \
       $(SRC_DIR)/training.c \
       $(SRC_DIR)/image_io.c \
       $(SRC_DIR)/main.c

# Object files
OBJS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRCS))

# Target binary
TARGET = $(BUILD_DIR)/cnn_framework

# =============================================================================
# Default target
# =============================================================================
.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "  LINK  $@"
	@$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "  Build complete: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo "  CC    $<"
	@$(CC) $(CFLAGS) -I$(INC_DIR) -c -o $@ $<

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# =============================================================================
# Run target
# =============================================================================
.PHONY: run
run: $(TARGET)
	@echo ""
	@echo "--- Running CNN Framework ---"
	@echo ""
	@./$(TARGET)

# =============================================================================
# Debug build
# =============================================================================
.PHONY: debug
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: clean $(TARGET)
	@echo "  Debug build complete (with AddressSanitizer)"

# =============================================================================
# Clean
# =============================================================================
.PHONY: clean
clean:
	@echo "  CLEAN"
	@rm -rf $(BUILD_DIR)
	@rm -f model.bin
