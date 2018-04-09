import Graph from "../../Graph";
import Expression from "../Expression";
import {ExpressionTypes} from "../ExpressionTypes";
import {Conv2dOptions} from "./Conv2d";

export default class Conv2dKernelGrad extends Expression {

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
    return this.kernel.shape;
  }

  get type() {
    return ExpressionTypes.Conv2dKernelGrad;
  }

  constructor(image: Expression, kernel: Expression, grad: Expression, options: Conv2dOptions, graph: Graph, name?: string) {
    super(graph, name);
    this._image = image;
    this._kernel = kernel;
    this._grad = grad;
    this._options = options;
  }

  buildSubExpression(): Expression {
    let xCol = this.image.im2col({
      kernelNum: this.kernel.shape[0],
      kernelChannel: this.kernel.shape[1],
      kernelHeight: this.kernel.shape[2],
      kernelWidth: this.kernel.shape[3],
      ...this.options
    });
    let gradReshape = this.grad.reshape([this.kernel.shape[0], -1]);
    let matmul = this.factory.matmul(gradReshape, xCol, false, true);
    return matmul.reshape(this.kernel.shape);
  }
}