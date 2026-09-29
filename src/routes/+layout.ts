// 纯静态站点：所有页面都预渲染，运行时不需要服务端
export const prerender = true;

// 纯静态托管（没有 `.html` 重写规则）下，`/licenses` 必须落成
// `licenses/index.html` 才能被直链访问——否则会 404。
export const trailingSlash = 'always';
