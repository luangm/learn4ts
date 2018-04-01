import Expression from "../expression/Expression";

export type VisitFunc = (node: Expression, params?: any) => any;

export default interface Visitor {

  readonly registry: Map<string, VisitFunc>;

  register(type: string, method: VisitFunc): void;

  visit(node: Expression, params?: any): void;

}