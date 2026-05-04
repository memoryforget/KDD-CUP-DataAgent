export declare class SSEParserTransform extends TransformStream<string, any> {
    private buffer;
    private currentEvent;
    constructor();
    private processLine;
}
