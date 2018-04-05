import Learn from "../../src/index";
import {Tensor} from "tensor4js";

test("params", function () {

  let X = Learn.variable([3], "X");
  let Y = Learn.variable([4], "Y");
  let W = Learn.parameter(Tensor.create(1), "weight");
  let b = Learn.parameter(Tensor.create([1, 2, 3]), "bias");

  let group1 = Learn.group([X, W, Y, b], "Group1");
  let group2 = Learn.group([X, b, Y, W], "Group2");
  let group3 = Learn.group([X, b, Y, W], "Group2");
  let group4 = Learn.group([X, b, Y, W], "Group4");
  let group5 = Learn.group([X, b, Y, W]);
  let group6 = Learn.group([X, b, Y, W]);

  console.log(group1.params);
  console.log(group2.params);
  console.log(group3.params);
  console.log(group4.params);
  console.log(group5.params);

  expect(group3).toBe(group2);
  expect(group3.id).toEqual(group2.id);
  expect(group4).not.toBe(group2);
  expect(group5).toBe(group6);
  expect(group5).not.toBe(group4);

  for (let [key, value] of group1.graph.nodes) {
    console.log(key, value.name);
  }
});