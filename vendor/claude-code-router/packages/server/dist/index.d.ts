interface RunOptions {
    port?: number;
    logger?: any;
}
declare function getServer(options?: RunOptions): Promise<any>;
export { getServer };
export type { RunOptions };
export type { IAgent, ITool } from "./agents/type";
export { initDir, initConfig, readConfigFile, writeConfigFile, backupConfigFile } from "./utils";
export { pluginManager, tokenSpeedPlugin } from "@musistudio/llms";
