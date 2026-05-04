/**rewriteStream
 * Read source readablestream and return a new readablestream, processor processes source data and pushes returned new value to new stream, no push if no return value
 * @param stream
 * @param processor
 */
export declare const rewriteStream: (stream: ReadableStream, processor: (data: any, controller: ReadableStreamController<any>) => Promise<any>) => ReadableStream;
