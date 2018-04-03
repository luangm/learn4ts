import Graph from "./Graph";
import Expression from "./expression/Expression";
import {Tensor} from "tensor4js";
import EvaluationVisitor from "./visitor/EvaluationVisitor";

export default class Session {

  private readonly _graph: Graph;
  private _stateMap: Map<number, boolean>;
  private _valueMap: Map<number, Tensor>;
  private _visitor: EvaluationVisitor;

  get graph() {
    return this._graph;
  }

  constructor(graph: Graph) {
    this._graph = graph;
    this._valueMap = new Map<number, Tensor>();
    this._stateMap = new Map<number, boolean>();
    this._visitor = new EvaluationVisitor(this);
  }

  // TODO: If somehow cannot find value, should return Tensor.Empty or something.
  eval(node: Expression): Tensor {
    this._visitor.visit(node);
    return this.getValue(node) || Tensor.create(0);
  }

  getValue(node: Expression): Tensor | undefined {
    return this._valueMap.get(node.id);
  }

  isValid(node: Expression): boolean {
    return this._stateMap.get(node.id) || false;
  }

  setValue(node: Expression, value: Tensor): void {
    this._valueMap.set(node.id, value);
    this._invalidateObservers(node);
    this._stateMap.set(node.id, true);
  }

  private _invalidateObservers(node: Expression) {
    for (let observer of node.observers) {
      // only need to trigger invalidation of nodes that are previously validated
      // Also helps to prevent circular reference infinite loop
      if (this.isValid(observer)) {
        this._stateMap.set(observer.id, false);
        this._invalidateObservers(observer);
      }
    }
  }
}