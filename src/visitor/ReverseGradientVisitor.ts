import Graph from "../Graph";
import Add from "../expression/binary/Add";
import Divide from "../expression/binary/Divide";
import MatMul from "../expression/binary/MatMul";
import Multiply from "../expression/binary/Multiply";
import Subtract from "../expression/binary/Subtract";
import Expression from "../expression/Expression";
import Reshape from "../expression/special/Reshape";
import Absolute from "../expression/transform/Absolute";
import Cosh from "../expression/transform/Cosh";
import Cosine from "../expression/transform/Cosine";
import Expm1 from "../expression/transform/Expm1";
import Exponential from "../expression/transform/Exponential";
import Log1p from "../expression/transform/Log1p";
import Logarithm from "../expression/transform/Logarithm";
import Negate from "../expression/transform/Negate";
import Reciprocal from "../expression/transform/Reciprocal";
import Relu from "../expression/transform/Relu";
import Sigmoid from "../expression/transform/Sigmoid";
import Sine from "../expression/transform/Sine";
import Sinh from "../expression/transform/Sinh";
import Sqrt from "../expression/transform/Sqrt";
import Square from "../expression/transform/Square";
import Tangent from "../expression/transform/Tangent";
import Tanh from "../expression/transform/Tanh";
import Visitor, {VisitFunc} from "./Visitor";
import {ExpressionTypes} from "../expression/ExpressionTypes";
import Elu from "../expression/transform/Elu";

export default class ReverseGradientVisitor implements Visitor {

    private _gradMap: Map<number, Expression[]>;
    private readonly _graph: Graph;
    private readonly _registry: Map<string, VisitFunc>;
    private _startId: number; // if not 0, the visit already started

    get factory() {
        return this.graph.factory;
    }

    get graph() {
        return this._graph;
    }

    get registry() {
        return this._registry;
    }

    constructor(graph: Graph) {
        this._graph = graph;
        this._registry = new Map<string, VisitFunc>();
        this.init();
    }

    register(type: string, method: VisitFunc): void {
        this.registry.set(type, method);
    }

    visit(node: Expression, params?: any): void {
        // initialize
        if (!this._startId) {
            this._startId = node.id;
            this._gradMap = new Map<number, Expression[]>();
        }

        // body
        let grad = this._graph.addNode(params || this.factory.fill(1, node.shape));
        this.addGradient(node, grad);

        let method = this.registry.get(node.type);
        if (method) {
            let grads = method(node, grad);

            for (let i = 0; i < node.dependencies.length; i++) {
                let dependency = node.dependencies[i];
                dependency.accept(this, grads[i]);
            }
        }

        // finalize
        if (this._startId === node.id) {
            this._startId = 0;
            for (let key of this._gradMap.keys()) {
                let list = this._gradMap.get(key);
                let addN = this.factory.addN(list);
                node.setGradient(key, addN);
            }
            this._gradMap = null;
        }
    }

    private addGradient(target: Expression, grad: Expression) {
        let list = this._gradMap.get(target.id) || [];
        list.push(grad);
        this._gradMap.set(target.id, list);
    }

    private init() {
        this.register(ExpressionTypes.Add, Add.gradients);
        this.register(ExpressionTypes.Divide, Divide.gradients);
        this.register(ExpressionTypes.MatMul, MatMul.gradients);
        // this.register(ExpressionTypes.Maximum, Maximum.gradients);
        // this.register(ExpressionTypes.Minimum, Minimum.gradients);
        // this.register(ExpressionTypes.Modulo, Modulo.gradients);
        this.register(ExpressionTypes.Multiply, Multiply.gradients);
        this.register(ExpressionTypes.Subtract, Subtract.gradients);

        // this.register(ExpressionTypes.Constant, Constant.gradients);
        // this.register(ExpressionTypes.Parameter, Parameter.gradients);

        // this.register(ExpressionTypes.ReduceSum, ReduceSum.gradients);

        // this.register(ExpressionTypes.Assign, Assign.gradients);
        // this.register(ExpressionTypes.Fill, Fill.gradients);

        this.register(ExpressionTypes.Absolute, Absolute.gradients);
        this.register(ExpressionTypes.Cosine, Cosine.gradients);
        this.register(ExpressionTypes.Cosh, Cosh.gradients);
        this.register(ExpressionTypes.Expm1, Expm1.gradients);
        this.register(ExpressionTypes.Exponential, Exponential.gradients);
        this.register(ExpressionTypes.Log1p, Log1p.gradients);
        this.register(ExpressionTypes.Logarithm, Logarithm.gradients);
        this.register(ExpressionTypes.Negate, Negate.gradients);
        this.register(ExpressionTypes.Reciprocal, Reciprocal.gradients);
        this.register(ExpressionTypes.Relu, Relu.gradients);
        this.register(ExpressionTypes.Elu, Elu.gradients);
        // this.register(ExpressionTypes.Round, Round.gradients);
        // this.register(ExpressionTypes.RSqrt, RSqrt.gradients);
        this.register(ExpressionTypes.Sigmoid, Sigmoid.gradients);
        // this.register(ExpressionTypes.SigmoidGrad, SigmoidGrad.gradients);
        // this.register(ExpressionTypes.Sign, Sign.gradients);
        this.register(ExpressionTypes.Sine, Sine.gradients);
        this.register(ExpressionTypes.Sinh, Sinh.gradients);
        // this.register(ExpressionTypes.Softmax, Softmax.gradients);
        // this.register(ExpressionTypes.SoftmaxGrad, SoftmaxGrad.gradients);
        this.register(ExpressionTypes.Sqrt, Sqrt.gradients);
        // this.register(ExpressionTypes.SqrtGrad, SqrtGrad.gradients);
        this.register(ExpressionTypes.Square, Square.gradients);
        // this.register(ExpressionTypes.Step, Step.gradients);
        this.register(ExpressionTypes.Tangent, Tangent.gradients);
        // this.register(ExpressionTypes.TangentGrad, TangentGrad.gradients);
        this.register(ExpressionTypes.Tanh, Tanh.gradients);

        this.register(ExpressionTypes.Reshape, Reshape.gradients);
    }

}