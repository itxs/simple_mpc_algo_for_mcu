# Basic MPC Algorithm for Embedded Optimal Temperature Control

This project provides a basic Model Predictive Control (MPC) algorithm designed for optimal temperature control in embedded systems. The algorithm employs a binary search technique and a simplified Second Order Plus Dead Time (SOPDT) process model, which effectively represents real thermal systems.

## Overview
- **Algorithm**: Binary search-based MPC
- **Model**: Reduced SOPDT (Second Order Plus Dead Time without ability to make ripples)
- **Conversion**: Must be translated to C code for firmware integration

## Model Calibration
To calibrate the SOPDT model for a specific heater-media-sensor system, use a least squares method or a similar algorithm on the measured step response data of the real system.

## Demo Application
The source code includes a demo application to experiment with various parameters.

![Program screenshot](program.png)
