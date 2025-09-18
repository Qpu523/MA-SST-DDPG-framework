#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Qingwen Pu. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import scienceplots
from sklearn.model_selection import GroupShuffleSplit

plt.style.use(['science', 'no-latex'])
font = {'family': 'Times New Roman', 'size': 12}
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


# --- Core interaction (actors/critics/buffer/loop). ---
# `simulate(prev_state, actions)` and `reward_cal(state, expert_state)` are assumed implemented elsewhere.

num_agents = 2
num_states = 6
act_dim = 2
gamma = 0.99
tau = 0.01
batch_size = 256
buffer_capacity = 10000

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x0=None):
        self.theta, self.mean, self.std_dev, self.dt = theta, mean, std_deviation, dt
        self.x_prev = np.zeros_like(self.mean) if x0 is None else x0

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

class ReplayBuffer:
    def __init__(self, capacity=buffer_capacity, batch_size=batch_size):
        self.capacity, self.batch_size = capacity, batch_size
        self.ptr = 0
        self.s0 = np.zeros((capacity, num_states), dtype=np.float32)
        self.s1 = np.zeros((capacity, num_states), dtype=np.float32)
        self.a0 = np.zeros((capacity, act_dim), dtype=np.float32)
        self.a1 = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r  = np.zeros((capacity, num_agents), dtype=np.float32)
        self.ns0 = np.zeros((capacity, num_states), dtype=np.float32)
        self.ns1 = np.zeros((capacity, num_states), dtype=np.float32)

    def add(self, s0, s1, a0, a1, r, ns0, ns1):
        i = self.ptr % self.capacity
        self.s0[i], self.s1[i] = s0, s1
        self.a0[i], self.a1[i] = a0, a1
        self.r[i] = r
        self.ns0[i], self.ns1[i] = ns0, ns1
        self.ptr += 1

    def sample(self):
        n = min(self.ptr, self.capacity)
        idx = np.random.choice(n, self.batch_size, replace=False)
        to_tensor = lambda x: tf.convert_to_tensor(x[idx])
        return (
            to_tensor(self.s0), to_tensor(self.s1),
            to_tensor(self.a0), to_tensor(self.a1),
            tf.cast(to_tensor(self.r), tf.float32),
            to_tensor(self.ns0), to_tensor(self.ns1),
        )


def build_actor():
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(act_dim)(x)
    m = tf.keras.Model(inp, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-4))
    return m

def build_critic():
    a1 = layers.Input(shape=(act_dim,))
    a2 = layers.Input(shape=(act_dim,))
    a = layers.Concatenate()([a1, a2])
    a = layers.Dense(16, activation="relu")(a)
    x = layers.Concatenate()([s_feat, a])
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(64, activation="relu")(x)
    q = layers.Dense(1)(x)
    m = tf.keras.Model([s_inp, a1, a2], q)
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-4))
    return m

def soft_update(target, source, tau):
    for tw, sw in zip(target.weights, source.weights):
        tw.assign(tau * sw + (1. - tau) * tw)

def policy(net, state, noise=None, clip=5.0):
    a = tf.squeeze(net(tf.expand_dims(state, 0))).numpy()
    if noise is not None: a = a * (1.0 + noise())
    return np.clip(a, -clip, clip)

# --- Instantiate agents ---
actors      = [build_actor() for _ in range(num_agents)]
critics     = [build_critic() for _ in range(num_agents)]
t_actors    = [build_actor() for _ in range(num_agents)]
t_critics   = [build_critic() for _ in range(num_agents)]
for i in range(num_agents):
    soft_update(t_actors[i], actors[i], 1.0)
    soft_update(t_critics[i], critics[i], 1.0)

buffer = ReplayBuffer()
noise0 = OUActionNoise(mean=np.zeros(act_dim), std_deviation=0.2*np.ones(act_dim))
noise1 = OUActionNoise(mean=np.zeros(act_dim), std_deviation=0.2*np.ones(act_dim))

# --- Training loop skeleton (replace `prev_state`, `expert_state` with your data sampling) ---
def train_step():
    s0, s1, a0, a1, r, ns0, ns1 = buffer.sample()

    with tf.GradientTape() as tape:
        ta0 = t_actors[0](ns0, training=True)
        ta1 = t_actors[1](ns1, training=True)
        y0 = tf.stop_gradient(r[:, 0:1] + gamma * t_critics[0]([ns0, ta0, ta1], training=True))
        q0 = critics[0]([s0, a0, a1], training=True)
        loss0 = tf.reduce_mean(tf.square(y0 - q0))
    g0 = tape.gradient(loss0, critics[0].trainable_variables)
    critics[0].optimizer.apply_gradients(zip(g0, critics[0].trainable_variables))

    with tf.GradientTape() as tape:
        ta0 = t_actors[0](ns0, training=True)
        ta1 = t_actors[1](ns1, training=True)
        y1 = tf.stop_gradient(r[:, 1:1+1] + gamma * t_critics[1]([ns1, ta0, ta1], training=True))
        q1 = critics[1]([s1, a0, a1], training=True)
        loss1 = tf.reduce_mean(tf.square(y1 - q1))
    g1 = tape.gradient(loss1, critics[1].trainable_variables)
    critics[1].optimizer.apply_gradients(zip(g1, critics[1].trainable_variables))

    with tf.GradientTape() as tape:
        pa0 = actors[0](s0, training=True)
        pa1 = actors[1](s1, training=True)
        q = critics[0]([s0, pa0, pa1], training=True)
        aloss0 = -tf.reduce_mean(q)
    ag0 = tape.gradient(aloss0, actors[0].trainable_variables)
    actors[0].optimizer.apply_gradients(zip(ag0, actors[0].trainable_variables))

    with tf.GradientTape() as tape:
        pa0 = actors[0](s0, training=True)
        pa1 = actors[1](s1, training=True)
        q = critics[1]([s1, pa0, pa1], training=True)
        aloss1 = -tf.reduce_mean(q)
    ag1 = tape.gradient(aloss1, actors[1].trainable_variables)
    actors[1].optimizer.apply_gradients(zip(ag1, actors[1].trainable_variables))

    for i in range(num_agents):
        soft_update(t_actors[i], actors[i], tau)
        soft_update(t_critics[i], critics[i], tau)

# --- Example episode roll-out (pseudo; plug in your data pipeline) ---
def run_episode(batch):
    for i in range(batch):
        # prev_state = [s0_vec, s1_vec]    # obtain from dataset or env
        # expert_state = [e0_vec, e1_vec]  # supervision target if needed
        # a0 = policy(actors[0], prev_state[0], noise0)
        # a1 = policy(actors[1], prev_state[1], noise1)
        # next_state = simulate(prev_state, [a0, a1])      # implement
        # rewards = reward_cal(next_state, expert_state)   # implement
        # buffer.add(prev_state[0], prev_state[1], a0, a1, rewards, next_state[0], next_state[1])
        pass
    if buffer.ptr >= batch_size:
        train_step()
