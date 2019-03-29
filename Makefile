SRCS=main.cpp mnist.cpp lif.cpp stdp.cpp snn.cpp
NAME=main

CXX=g++
STRIP=strip

CXXFLAGS=-Wall -Wextra -Wno-unused-parameter -O2 -march=native
LDFLAGS=
LDDEPS=
DEFS=

INCLUDES=

OBJDIR=.objs
DEPDIR=.deps

OBJS=$(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))
DEPS=$(addprefix $(DEPDIR)/, $(SRCS:.cpp=.depends))

.PHONY : clean

all : $(NAME)

$(NAME) : $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(STARTUP) $^ $(LDDEPS) -o $@
	$(STRIP) $@

$(OBJDIR)/%.o : %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEFS) -c $< -o $@

$(OBJDIR) :
	mkdir -p $(OBJDIR)

$(OBJS) : | $(OBJDIR)

$(DEPDIR)/%.depends : %.cpp
	$(CXX) -M -MT"$(OBJDIR)/$(<:.cpp=.o)" $(CXXFLAGS) $(INCLUDES) $(DEFS) $< > $@

$(DEPDIR) :
	mkdir -p $(DEPDIR)

$(DEPS) : | $(DEPDIR)

clean :
	rm -f $(NAME)
	rm -rf $(OBJDIR) $(DEPDIR)

-include $(DEPS)
