"use strict";
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { ProgressView } = require("../progress");
test("stage labels, stderr bars and completion dedup match Java/Go", () => {
  const status = [],
    bars = [];
  const view = new ProgressView(
    "sample.pdf",
    true,
    "bar",
    { write: (s) => status.push(s) },
    { write: (s) => bars.push(s) },
  );
  const event = {
    stage: "table",
    completed: 1,
    total: 1,
    elapsed_seconds: 0.16,
  };
  view.event(event);
  view.event({
    stage: "assemble",
    completed: 0,
    total: 14,
    elapsed_seconds: 0,
  });
  for (let i = 0; i < 14; i++) view.event(event);
  view.finish();
  assert.equal(bars.join("").match(/Table Predict/g).length, 1);
  assert.match(
    bars.join(""),
    /\r.*Table Predict: 100% .*1\/1 \[processing 0.16s\]/,
  );
  assert.match(status.join(""), /completed in/);
});
