#!/usr/bin/env bash
find src include -name *.cpp -or -name *.h | xargs clang-format -i -sort-includes -style=Mozilla
