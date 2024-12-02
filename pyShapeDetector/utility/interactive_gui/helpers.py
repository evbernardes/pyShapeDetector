#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def get_pretty_name(func):
    return func.__name__.replace("_", " ").capitalize()


def parse_parameters_as_kwargs(parameters):
    return {name: value["default"] for name, value in parameters.items()}
