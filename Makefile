DIRGUARD = @mkdir -p $(@D)

CFLAGS = -Wall -Wextra -Werror
CXX = mpicxx
CXXFLAGS = -pedantic -std=c++17

UNAME = $(shell uname)
ifeq ($(UNAME), Darwin)
	CXXFLAGS += -ld_classic
endif

.PHONY: all
all: bin/main

bin/main: src/main.cpp
	$(DIRGUARD)
	$(CXX) $(CFLAGS) $(CXXFLAGS) -o $@ $<

.PHONY: clean
.SILENT: clean
clean:
	rm -f $(CXXEXEC)