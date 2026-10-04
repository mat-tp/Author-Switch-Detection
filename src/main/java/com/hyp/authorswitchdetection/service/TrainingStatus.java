package com.hyp.authorswitchdetection.service;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** Snapshot of the current/last training run, polled by the Train page via AJAX. */
public class TrainingStatus {
    public volatile boolean running = false;
    public volatile boolean done = false;
    public volatile int percent = 0;
    public volatile String currentTask = "";
    public volatile String error = null;
    public final List<String> log = Collections.synchronizedList(new ArrayList<>());

    public void reset() {
        running = false;
        done = false;
        percent = 0;
        currentTask = "";
        error = null;
        log.clear();
    }
}
