import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import {Conv2dOptions} from "./Conv2d";

export default class Conv2dImageGrad extends Expression {

  private readonly _grad: Expression;
  private readonly _image: Expression;
  private readonly _kernel: Expression;
  private readonly _options: Conv2dOptions;

  get dependencies() {
    return [this.image, this.kernel, this.grad];
  }

  get grad() {
    return this._grad;
  }

  get image() {
    return this._image;
  }

  get kernel() {
    return this._kernel;
  }

  get options() {
    return this._options;
  }

  get params() {
    return {
      type: this.type,
      name: this.name,
      image: this.image.id,
      kernel: this.kernel.id,
      grad: this.grad.id,
      options: this.options
    };
  }

  get shape(): number[] {
    return this.image.shape;
  }

  get type() {
    return ExpressionTypes.Conv2dImageGrad;
  }

  constructor(image: Expression, kernel: Expression, grad: Expression, options: Conv2dOptions, graph: Graph, name?: string) {
    super(graph, name);
    this._image = image;
    this._kernel = kernel;
    this._grad = grad;
    this._options = options;
  }

  buildSubExpression(): Expression {
    let numKernels = this.kernel.shape[0];
    let gradReshaped = this.grad.reshape([numKernels, -1]);
    let kReshaped = this.kernel.reshape([numKernels, -1]);
    let col = this.factory.matmul(kReshaped, gradReshaped, true, false);
    let im = col.col2im({
      imageNum: this.image.shape[0],
      imageChannel: this.image.shape[1],
      imageHeight: this.image.shape[2],
      imageWidth: this.image.shape[3],
      kernelHeight: this.kernel.shape[2],
      kernelWidth: this.kernel.shape[3],
      ...this.options
    });
    return im.reshape(this.image.shape);
  }
}