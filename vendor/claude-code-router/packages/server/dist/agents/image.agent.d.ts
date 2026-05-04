import { IAgent, ITool } from "./type";
export declare class ImageAgent implements IAgent {
    name: string;
    tools: Map<string, ITool>;
    constructor();
    shouldHandle(req: any, config: any): boolean;
    appendTools(): void;
    reqHandler(req: any, config: any): void;
}
export declare const imageAgent: ImageAgent;
