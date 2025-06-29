class PokerStrategy:
    def __init__(self):
        self.aggressiveness = 0.8  # 提高进攻性系数
        self.rank_values = {
            '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
            '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
        }
        # 阶段调整因子 - 提高翻牌前的进攻性
        self.stage_factors = {
            'preflop': 1.4,  # 翻牌前阶段更激进
            'flop': 1.1,
            'turn': 1.0,
            'river': 0.9      # 河牌阶段相对保守
        }
    
    def decide_action(self, state):
        # 确定当前阶段
        stage = self._determine_stage(state)
        
        # 1. 精确评估手牌强度
        hand_strength = self._evaluate_hand_strength(
            state['private_card'], 
            state['public_card'],
            stage
        )
        
        # 2. 计算位置优势
        position_factor = self._position_advantage(
            state['position'], 
            len(state['players'])
        )
        
        # 3. 评估对手行为
        opponent_aggression = self._assess_opponents(state['players'])
        
        # 4. 计算当前需要跟注的金额
        call_amount = self._calculate_call_amount(state)
        
        # 5. 综合决策
        return self._select_action(
            state['legal_actions'], 
            hand_strength,
            position_factor,
            opponent_aggression,
            state.get('raise_range', []),
            stage,
            call_amount
        )
    
    def _determine_stage(self, state):
        """根据公共牌数量确定游戏阶段"""
        num_public = len(state['public_card'])
        if num_public == 0:
            return 'preflop'
        elif num_public == 3:
            return 'flop'
        elif num_public == 4:
            return 'turn'
        else:  # 5
            return 'river'
    
    def _evaluate_hand_strength(self, private_cards, public_cards, stage):
        """精确评估手牌强度 (0-1)，基于完整牌型分析"""
        all_cards = private_cards + public_cards
        if stage == 'preflop':
            # 翻牌前阶段使用简化评估
            return self._evaluate_preflop_hand(private_cards)
        elif stage == 'flop' and len(all_cards) < 5:
            # 翻牌阶段考虑潜力
            return self._evaluate_flop_hand(private_cards, public_cards)
        else:
            # 转牌和河牌阶段，精确评估所有可能的5张牌组合
            return self._evaluate_full_hand(all_cards)
    
    def _evaluate_preflop_hand(self, private_cards):
        """翻牌前阶段的手牌评估"""
        # 1. 解析手牌
        rank1, suit1 = private_cards[0][0], private_cards[0][1]
        rank2, suit2 = private_cards[1][0], private_cards[1][1]
        val1 = self.rank_values[rank1]
        val2 = self.rank_values[rank2]
        
        # 计算最大值和最小值
        max_val = max(val1, val2)
        min_val = min(val1, val2)
        gap = max_val - min_val
        
        # 2. 基础强度计算
        strength = 0.0
        
        # 对子
        if rank1 == rank2:
            strength = 0.35 + (val1 - 2) * 0.04  # 提高对子价值
        
        # 同花
        elif suit1 == suit2:
            # 高牌同花
            if gap == 1:  # 连牌
                strength = 0.55 + min_val * 0.025  # 提高连牌价值
            elif gap == 2:
                strength = 0.45 + min_val * 0.025
            else:
                strength = 0.35 + max_val * 0.03  # 提高高牌价值
        
        # 连牌
        elif gap == 1:
            strength = 0.45 + min_val * 0.025  # 提高连牌价值
        elif gap == 2:
            strength = 0.4 + min_val * 0.025
        
        # 其他手牌
        else:
            strength = max_val * 0.025 - gap * 0.01  # 提高高牌价值
        
        # 提高小对子和高牌的价值
        if max_val >= 10 and min_val >= 8:  # 高牌如KQ, AJ等
            strength = max(strength, 0.45)
        elif rank1 == 'A' or rank2 == 'A':  # 任何带A的手牌
            strength = max(strength, 0.3)
        
        return min(max(strength, 0.05), 0.95)
    
    def _evaluate_flop_hand(self, private_cards, public_cards):
        """翻牌阶段的手牌评估，考虑潜力"""
        all_cards = private_cards + public_cards
        
        # 1. 基础强度
        base_strength = self._evaluate_full_hand(all_cards)
        
        # 2. 潜力评估
        flush_potential = self._flush_potential(private_cards, public_cards)
        straight_potential = self._straight_potential(private_cards, public_cards)
        
        # 3. 综合强度 = 基础强度 + 潜力加成
        potential_bonus = (flush_potential + straight_potential) * 0.15
        return min(base_strength + potential_bonus, 0.95)
    
    def _flush_potential(self, private_cards, public_cards):
        """评估同花潜力"""
        suits = [card[1] for card in private_cards + public_cards]
        suit_counts = {suit: suits.count(suit) for suit in set(suits)}
        
        # 计算最佳同花潜力
        max_count = max(suit_counts.values())
        if max_count >= 4:
            return 1.0  # 已经有4张同花
        elif max_count == 3:
            return 0.6  # 有3张同花
        elif max_count == 2 and any(s == private_cards[0][1] for s in [private_cards[1][1]]):
            return 0.3  # 手牌同花
        return 0.0
    
    def _straight_potential(self, private_cards, public_cards):
        """评估顺子潜力"""
        all_ranks = sorted([self.rank_values[card[0]] for card in private_cards + public_cards])
        gaps = []
        
        # 计算最大间隔
        for i in range(1, len(all_ranks)):
            gap = all_ranks[i] - all_ranks[i-1]
            if gap > 1:
                gaps.append(gap)
        
        # 评估潜力
        if not gaps:
            return 1.0  # 已经是顺子
        elif max(gaps) <= 2:
            return 0.7  # 开间隔顺子听牌
        elif max(gaps) <= 3:
            return 0.4  # 双头顺子听牌
        return 0.0
    
    def _evaluate_full_hand(self, all_cards):
        """精确评估完整牌型（5-7张牌）的强度"""
        # 生成所有可能的5张牌组合
        from itertools import combinations
        best_score = 0
        
        for combo in combinations(all_cards, 5):
            score = self._evaluate_five_card_hand(list(combo))
            if score > best_score:
                best_score = score
        
        # 返回最佳牌型的强度
        return best_score
    
    def _evaluate_five_card_hand(self, five_cards):
        """评估5张牌的牌型强度"""
        # 解析牌面和花色
        ranks = [card[0] for card in five_cards]
        suits = [card[1] for card in five_cards]
        values = sorted([self.rank_values[r] for r in ranks], reverse=True)
        
        # 检查同花
        is_flush = len(set(suits)) == 1
        
        # 检查顺子
        is_straight = False
        straight_high = 0
        # 标准顺子
        for i in range(len(values) - 4):
            if values[i] - values[i+4] == 4 and len(set(values[i:i+5])) == 5:
                is_straight = True
                straight_high = values[i]
                break
        # A-5顺子特殊情况
        if not is_straight and set(values) == {14, 2, 3, 4, 5}:
            is_straight = True
            straight_high = 5
        
        # 同花大顺（皇家同花顺）
        if is_flush and is_straight and straight_high == 14:
            return 1.0
        
        # 同花顺
        if is_flush and is_straight:
            strength = 0.9 + (straight_high - 5) * 0.01
            return min(strength, 0.99)
        
        # 检查四条
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        if 4 in rank_counts.values():
            quad_rank = [r for r, count in rank_counts.items() if count == 4][0]
            kicker = max([self.rank_values[r] for r in ranks if r != quad_rank])
            strength = 0.8 + (self.rank_values[quad_rank] - 2) * 0.01 + kicker * 0.001
            return min(strength, 0.89)
        
        # 检查葫芦
        if 3 in rank_counts.values() and 2 in rank_counts.values():
            trips_rank = [r for r, count in rank_counts.items() if count == 3][0]
            pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
            strength = 0.7 + (self.rank_values[trips_rank] - 2) * 0.01 + self.rank_values[pair_rank] * 0.001
            return min(strength, 0.79)
        
        # 检查同花
        if is_flush:
            # 取五张最高牌
            flush_ranks = sorted([self.rank_values[r] for r in ranks], reverse=True)[:5]
            strength = 0.6 + sum(r * (0.1 ** (i+1)) for i, r in enumerate(flush_ranks))
            return min(strength, 0.69)
        
        # 检查顺子
        if is_straight:
            strength = 0.5 + (straight_high - 5) * 0.01
            return min(strength, 0.59)
        
        # 检查三条
        if 3 in rank_counts.values():
            trips_rank = [r for r, count in rank_counts.items() if count == 3][0]
            kickers = sorted([self.rank_values[r] for r in ranks if r != trips_rank], reverse=True)[:2]
            strength = 0.4 + (self.rank_values[trips_rank] - 2) * 0.01 + sum(k * 0.001 for k in kickers)
            return min(strength, 0.49)
        
        # 检查两对
        pairs = [r for r, count in rank_counts.items() if count == 2]
        if len(pairs) >= 2:
            pair_ranks = sorted([self.rank_values[r] for r in pairs], reverse=True)[:2]
            kicker = max([self.rank_values[r] for r in ranks if r not in pairs])
            strength = 0.3 + sum(p * 0.01 for p in pair_ranks) + kicker * 0.001
            return min(strength, 0.39)
        
        # 检查一对
        if 2 in rank_counts.values():
            pair_rank = [r for r, count in rank_counts.items() if count == 2][0]
            kickers = sorted([self.rank_values[r] for r in ranks if r != pair_rank], reverse=True)[:3]
            strength = 0.2 + (self.rank_values[pair_rank] - 2) * 0.01 + sum(k * 0.001 for k in kickers)
            return min(strength, 0.29)
        
        # 散牌
        high_cards = sorted(values, reverse=True)[:5]
        strength = 0.1 + sum(c * (0.1 ** (i+1)) for i, c in enumerate(high_cards))
        return min(strength, 0.19)
    
    def _position_advantage(self, position, num_players):
        """计算位置优势 (0-1)"""
        # 位置越靠后优势越大
        return position / (num_players - 1) if num_players > 1 else 0.5
    
    def _assess_opponents(self, players):
        """评估对手进攻性 (0-1)"""
        # 分析对手下注模式
        total_contrib = sum(p.get('contribution', 0) for p in players)
        max_contrib = max(p.get('contribution', 0) for p in players) if players else 0
        avg_contrib = total_contrib / len(players) if players else 0
        
        # 计算进攻性：最高下注与平均下注的比例
        if avg_contrib > 0:
            aggression = min(max_contrib / avg_contrib, 3.0) / 3.0
        else:
            aggression = 0.0
            
        return aggression
    
    def _calculate_call_amount(self, state):
        """计算当前需要跟注的金额"""
        current_player = None
        for player in state['players']:
            if player['position'] == state['position']:
                current_player = player
                break
        
        if not current_player:
            return 0
        
        # 当前玩家已下注金额
        current_bet = current_player['contribution']
        # 当前最大下注金额
        max_bet = max(p['contribution'] for p in state['players'])
        
        return max(0, max_bet - current_bet)
    
    def _select_action(self, legal_actions, hand_strength, position_factor, opponent_aggression, raise_range, stage, call_amount):
        """基于参数选择最佳行动 - 重点优化raise策略"""
        # 获取阶段调整因子
        stage_factor = self.stage_factors.get(stage, 1.0)
        
        # 计算综合决策分数（考虑阶段因子）
        decision_score = (hand_strength * 0.5 + 
                         position_factor * 0.2 - 
                         opponent_aggression * 0.2 +  # 降低对手进攻性影响
                         self.aggressiveness * 0.3) * stage_factor  # 提高进攻性影响
        
        decision_score = max(0, min(1, decision_score))
        
        # 翻牌后阶段提高决策分数，减少弃牌倾向
        if stage != 'preflop':
            # 已下注较大时进一步减少弃牌倾向
            if call_amount > 0:  # 当前需要跟注
                decision_score *= 1.2  # 提高20%决策分数
            else:
                decision_score *= 1.1  # 提高10%决策分数
        
        # 河牌阶段特殊处理：大幅降低弃牌倾向
        if stage == 'river':
            decision_score = min(1.0, decision_score * 1.25)
        
        # 翻牌前阶段特殊处理：增加raise可能性
        if stage == 'preflop':
            # 好牌时主动加注
            if hand_strength > 0.5 and 'raise' in legal_actions:
                if raise_range:
                    min_raise, max_raise = raise_range
                    # 中等牌力适度加注
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.4)
                    return f'r{raise_amount}'
                else:
                    return 'raise'
            
            # 非常弱的牌在翻牌前也可以弃牌
            if hand_strength < 0.15 and 'fold' in legal_actions:
                return 'fold'
            # 有check选项时优先check
            elif 'check' in legal_actions:
                return 'check'
            # 对手下注高时，只有好牌才跟注
            elif 'call' in legal_actions:
                # 高额下注：超过初始筹码的5%
                high_bet = call_amount > 1000
                if high_bet:
                    if hand_strength > 0.4:  # 只有好牌才跟高额下注
                        return 'call'
                    else:
                        return 'fold' if 'fold' in legal_actions else 'check'
                else:
                    return 'call'  # 小额下注尽量跟注
            # 其他情况
            else:
                return 'fold' if 'fold' in legal_actions else 'check'
        
        # 其他阶段
        # 根据牌型强度调整决策
        if hand_strength > 0.7:  # 降低强牌阈值
            decision_score *= 1.3  # 提高强牌决策分数
        elif hand_strength > 0.5:  # 中等强牌
            decision_score *= 1.2
        elif hand_strength < 0.2:  # 弱牌
            decision_score *= 0.8  # 降低弱牌决策分数影响
        
        # 根据对手进攻性调整
        if opponent_aggression > 0.7 and hand_strength < 0.6:  # 对手激进且我方牌弱
            decision_score *= 0.8  # 适度降低分数
        
        # 确保分数在合理范围内
        decision_score = max(0.05, min(0.95, decision_score))
        
        # 对手下注高时，好牌时反加
        high_bet = call_amount > 2000
        if high_bet:
            # 翻牌后阶段，已下注较大时优先跟注而非弃牌
            if call_amount > 500:  # 已下注较大
                # 中等牌力优先跟注
                if hand_strength >= 0.4 and 'call' in legal_actions:
                    return 'call'
                
                # 考虑底池赔率：当跟注金额/潜在收益 < 0.3时跟注
                if hand_strength > 0.3 and 'call' in legal_actions:
                    # 估算潜在收益（底池金额 + 对手可能跟注的额外金额）
                    pot_odds = call_amount / (call_amount * 3 + 1000)  # 简化计算
                    if pot_odds < 0.3:  # 有利的底池赔率
                        return 'call'
            
            # 河牌阶段更积极跟注
            if stage == 'river' and hand_strength > 0.35 and 'call' in legal_actions:
                return 'call'
            
            # 非常强的牌反加
            if hand_strength > 0.6 and 'raise' in legal_actions:  # 降低反加阈值
                min_raise, max_raise = raise_range
                # 根据牌型强度决定加注大小
                if hand_strength > 0.8:
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.9)  # 强牌更大加注
                else:
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.6)  # 中等牌力适度加注
                return f'r{raise_amount}'
            
            # 牌力较弱但已下注较大时，优先跟注而非弃牌
            if call_amount > 500 and hand_strength > 0.3 and 'call' in legal_actions:
                return 'call'
        
        # 主动加注策略 - 降低raise阈值
        if 'raise' in legal_actions and decision_score > 0.4:  # 降低决策分数阈值
            if raise_range:
                min_raise, max_raise = raise_range
                # 根据牌型强度决定加注大小
                if hand_strength > 0.8:
                    # 强牌激进加注
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.9)
                elif hand_strength > 0.6:
                    # 中等牌力适度加注
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.6)
                else:
                    # 弱牌保守加注
                    raise_amount = min_raise + int((max_raise - min_raise) * 0.3)
                return f'r{raise_amount}'
            else:
                return 'raise'
            
        elif 'call' in legal_actions and decision_score > 0.3:  # 降低call阈值
            return 'call'
            
        elif 'check' in legal_actions and decision_score > 0.2:
            return 'check'
            
        else:
            # 翻牌后阶段尽量避免弃牌
            if stage != 'preflop':
                # 已下注较大时优先跟注
                if call_amount > 500 and 'call' in legal_actions:
                    return 'call'
                # 有check选项时优先check
                elif 'check' in legal_actions:
                    return 'check'
            
            # 最终决策
            return 'fold' if 'fold' in legal_actions else 'check'