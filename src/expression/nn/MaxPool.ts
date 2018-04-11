import {ShapeUtils} from "tensor4js";
import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";

export interface MaxPoolOptions {
  kernelChannel: number;
  kernelHeight: number;
  kernelNum: number;
  kernelWidth: number;
  padHeight?: number;
  padWidth?: number;
  strideHeight?: number;
  strideWidth?: number;
}

export default class MaxPool extends Expression {

  private readonly _image: Expression;
  private readonly _options: MaxPoolOptions;
  private readonly _shape: number[];

  get dependencies() {
    return [this.image];
  }

  get image() {
    return this._image;
  }

  get options() {
    return this._options;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      image: this.image.id,
      options: this.options
    };
  }

  get shape(): number[] {
    return this._shape;
  }

  get type() {
    return ExpressionTypes.MaxPool;
  }

  constructor(image: Expression, options: MaxPoolOptions, graph: Graph, name?: string) {
    super(graph, name);
    this._image = image;
    this._options = options;
    let kernelShape = [options.kernelNum, options.kernelChannel, options.kernelHeight, options.kernelWidth];
    this._shape = ShapeUtils.computeConv2dShape(image.shape, kernelShape, options);
  }

  // static gradients(expression: Expression, grad: Expression): Expression[] {
  //   let node = expression as MaxPool;
  //   let imageGrad = node.factory.conv2dImageGrad(node.image, node.kernel, grad, node.options);
  //   let kernelGrad = node.factory.conv2dKernelGrad(node.image, node.kernel, grad, node.options);
  //   return [imageGrad, kernelGrad];
  // }
  //
  // buildInternal(): Expression {
  //   let imageShape = this.image.shape;
  //   let kernelShape = this.kernel.shape;
  //   let outputShape = ShapeUtils.computeConv2dShape(imageShape, kernelShape, this.options);
  //   let xCol = this.image.im2col({
  //     kernelNum: kernelShape[0],
  //     kernelChannel: kernelShape[1],
  //     kernelHeight: kernelShape[2],
  //     kernelWidth: kernelShape[3],
  //     ...this.options
  //   });
  //
  //   let kRows = this.kernel.reshape([kernelShape[0], -1]);
  //   let result = kRows.matmul(xCol);
  //   let reshaped = result.reshape([kernelShape[0], imageShape[0], outputShape[2], outputShape[3]]);
  //   return reshaped.transpose([1, 0, 2, 3]);
  // }
}